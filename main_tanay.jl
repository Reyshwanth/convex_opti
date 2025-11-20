# The code follows the logic and flow that is borrowed from the paper on Q-Learning by Chirag et. al.

using LibGEOS, Graphs, Random, StaticArrays, BSplineKit, LinearAlgebra, ForwardDiff
using JuMP
using Gurobi

# to keep the entire environment in one object.
struct Environment
    boundary::GEOSGeometry
    obstacles::Vector{GEOSGeometry}
end

# ============== ALGORITHM 1 ================

# load the environment from the 
# took help from online sources and chat gpt to write this function
function load_environment(path::String)
    txt = read(path, String)
    polys = [] #array to store GEOSGeometry polygons
    for block in split(txt, "\n\n")
        pts = []  #array to store tuples (x, y)
        for line in split(block, '\n')
            s = strip(line)  #remove whitespace
            isempty(s) && continue
            # skip headers if any
            if startswith(s, "poly") continue end
            x,y = parse.(Float64, split(s, ','))
            push!(pts, (x,y))
        end
        if length(pts) >= 3   #polygon must have at least 3 vertices
            coords = vcat([(p[1], p[2]) for p in pts]...)
            push!(polys, geometry_from_coords(coords))
        end
    end

    # choose largest polygon area as boundary
    areas = map(geos_area, polys)
    idx = argmax(areas)
    boundary = polys[idx]
    obstacles = [polys[i] for i in eachindex(polys) if i != idx]
    Environment(boundary, obstacles)
end


#This function converts [(x1, y1) , (x2, y2), (x3, y3), ...] to flattened format [x1, y1, x2, y2, x3, y3, ...]
function geometry_from_coords(coords)
    arr = reduce(vcat, [[c[1], c[2]] for c in coords])
    return GEOSGeom(create_polygon(arr))
end

#helper functions
geos_area(g::GEOSGeometry) = area(g)
function point_geom(p::Vec2)
    GEOSGeom(create_point((p[1], p[2])))
end
function seg_geom(a::Vec2, b::Vec2)
    arr = [a[1], a[2], b[1], b[2]]
    GEOSGeom(create_line_string(arr))
end

#min distance from a point to all obstacles 
function min_dist_to_obstacles(pt::Vec2, env::Environment)
    pgeom = point_geom(pt)
    dists = map(o->distance(pgeom, o), env.obstacles)
    isempty(dists) ? Inf : minimum(dists)
end


# Waypoint based graph generation

# add obstacle vertices + grid samples
function generate_waypoints(env::Environment, start::Vec2, goal::Vec2; grid_res=0.5, clearance=0.3)
    # collect obstacle vertices
    W = Vector{Vec2}()
    push!(W, start); push!(W, goal)
    for o in env.obstacles
        coords = coordinates(o)  # returns Nx2 array
        for r in eachrow(coords)
            push!(W, Vec2(r[1], r[2]))
        end
    end
    # bounding box from boundary
    xmin, ymin, xmax, ymax = envelope_bounds(env.boundary)
    for x in xmin:grid_res:xmax
        for y in ymin:grid_res:ymax
            p = Vec2(x,y)
            # inside and at least clearance from obstacles
            if contains_point(env.boundary, p) && min_dist_to_obstacles(p, env) >= clearance
                push!(W, p)
            end
        end
    end
    W_unique = unique(W)  # deduplicate
    return W_unique
end

function envelope_bounds(g::GEOSGeometry)
    env = envelope(g)
    coords = coordinates(env)
    xs = coords[:,1]; ys = coords[:,2]
    return minimum(xs), minimum(ys), maximum(xs), maximum(ys)
end

function contains_point(poly::GEOSGeometry, p::Vec2)
    pg = point_geom(p)
    contains(poly, pg) || within(pg, poly)
end

# build visibility graph: edges if segment does not intersect obstacles and inside boundary
function build_visibility_graph(W::Vector{Vec2}, env::Environment; clearance=0.0)
    n = length(W)
    g = SimpleGraph(n)
    for i in 1:n-1
        for j in i+1:n
            a, b = W[i], W[j]
            seg = seg_geom(a,b)
            # check inside boundary
            if !within(seg, env.boundary) && !covered_by(seg, env.boundary)
                continue
            end
            # check clearance w.r.t obstacles (segment buffer)
            ok = true
            for o in env.obstacles
                if intersects(seg, o)
                    ok = false; break
                end
                # option: also ensure minimum distance by buffering segment
                if clearance > 0
                    if distance(seg, o) < clearance
                        ok = false; break
                    end
                end
            end
            if ok
                add_edge!(g, i, j)
            end
        end
    end
    return g
end