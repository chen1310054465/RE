$(document).ready(function () {
    var t = !1, n = 0, s = Math.PI, o = 2 * s, e = {0: "#1aaca4", 1: "#5d3fd5"}, r = window.innerWidth,
        a = window.innerHeight, m = r, h = a - 10,
        i = d3.select(".kg-graph").select(".navigation-bar").style("visibility", "hidden"), l = h - 200;
    i.style("height", l + "px");
    var c = d3.select(".kg-graph").select(".tooltip"), d = d3.select(".kg-graph").select(".legend");
    d.style("visibility", "hidden"), d.select(".searchbar").on("change", function () {
        var t = d3.event.target.value, e = t.split(":");
        null == t || 0 == t.length ? x.select("#relation-container").select(".current").selectAll(".relation").transition().duration(500).style("opacity", 1) : 1 == e.length ? x.select("#relation-container").select(".current").selectAll(".relation").transition().duration(500).style("opacity", function (t) {
            return t.name == e[0] ? 1 : 0
        }) : 2 == e.length && "p" == e[0] && x.select("#relation-container").select(".current").selectAll(".relation").transition().duration(500).style("opacity", function (t) {
            return t.predicate == e[1] ? 1 : 0
        })
    });
    var u = !0, p = 0, f = 100, g = !1, y = !1, v = d3.select("svg");
    v.attr("width", m), v.attr("height", h), v.style("margin-top", 5);
    var x = v.append("g").attr("class", "canvas").attr("transform", "translate(" + m / 2 + "," + h / 2 + ")");
    x.append("g").attr("id", "spiral-container"), x.append("g").attr("id", "relation-container"), x.append("g").attr("id", "node-container"), x.append("g").attr("id", "concept-container");
    var b = d3.arc().innerRadius(23).outerRadius(25).startAngle(Math.PI / 180 * 45).endAngle(3),
        w = x.append("path").style("fill", "#605c62").attr("d", b).attr("id", "loadingCurve").attr("visibility", "hidden"),
        A = d3.zoom().on("zoom", function () {
            if (d3.event.sourceEvent.deltaY && 1 == n) {
                var t = parseInt(d3.event.sourceEvent.deltaY);
                t < -20 && (t = -20), 20 < t && (t = 20), (f += t) < 1 && (f = 1), 110 < f && (f = 110)
            }
        });
    v.call(A);
    for (var M = [], j = window.location.href.split("?")[1].split("&"), X = "", Y = !1, T = 0; T < j.length; T++) {
        var D = j[T].split("=");
        "entity" == D[0] ? X = D[1] : "lock" == D[0] && (Y = !0)
    }

    function E(t, e) {
        d.style("visibility", "visible"), d3.select(".kg-graph").select(".predicate-filter").style("visibility", "visible"), n = 1, M = [], x.select("#node-container").selectAll(".node").data(M).exit().remove(), x.select("#concept-container").remove(), x.select("#spiral-container").remove(), N(M, {
            name: t.property.name,
            image: t.property.pic,
            x: 0,
            y: 0,
            relations: e
        }), F(), i.style("visibility", "visible")
    }

    function I() {
        var t = M[M.length - 1];
        !function (t, n) {
            var r = t.relations;
            r.forEach(function (t, e) {
                t._graphid = n + "-" + e
            });
            for (var e = [], a = 0; a < r.length; a++) e.push(d3.randomUniform(3, 5)());
            var i = d3.sum(e), l = e.map(function (t) {
                return t / i * o
            }), c = [];
            l.reduce(function (t, e, n) {
                return c[n] = t + e
            }, 0), cur = 0, sibs = r, r = [], $.each(c, function (t, e) {
                rr = Math.random() / 5 + .15, k = (cur + Math.round(rr * (sibs.length - 1))) % sibs.length, cur = k, sibs[cur].angle = e, r[t] = sibs[cur], sibs.splice(cur, 1)
            }), $.each(r, function (t, e) {
                sibs = [];
                for (var n = -3; n < 4; n++) 0 != n && (cur = (t + n + r.length) % r.length, tmp = {
                    angle: e.angle - r[cur].angle,
                    distance: r[cur].distance
                }, 0 != tmp.angle && sibs.push(tmp));
                e.sibs = sibs
            }), t.relations = r
        }(t, M.length - 1);
        var e = x.select("#relation-container").select(".current").selectAll(".relation").data(function (t) {
                return t.relations
            }).enter().append("g").attr("class", "relation"),
            n = (e.append("line").attr("class", "line").attr("x1", "0").attr("y1", "0").attr("x2", "0").attr("y2", "0").style("stroke", function (t) {
                return B(t)
            }).style("stroke-width", 2).on("mouseover", function (t) {
                g || (u = !1, t.context && (c.select(".title").text("潜在关系解释"), c.select(".des").text(t.context), c.select(".result-img a").style("background-image", "url('" + t.image + "')"), c.transition().duration(200).style("opacity", 1)))
            }).on("mouseout", function (t) {
                g || (u = !0, c.transition().duration(200).style("opacity", 0))
            }).on("mousemove", function (t) {
                g || (1e3 < m ? (d3.event.pageX > m / 2 + 50 ? c.style("left", d3.event.pageX - 20 - 520 + "px") : d3.event.pageX < m / 2 - 50 ? c.style("left", d3.event.pageX + 10 + "px") : c.style("left", d3.event.pageX - 250 + "px"), d3.event.pageY > h / 2 ? c.style("top", d3.event.pageY - 150 + "px") : c.style("top", d3.event.pageY + 30 + "px")) : c.style("position", "fixed").style("width", "400").style("height", "100").style("left", "calc(50% - 200px)").style("top", "calc(50% - 50px)").style("display", "block"))
            }), e.append("g").attr("class", "objectAndText")),
            r = (n.append("defs").attr("id", "imgdefs").append("pattern").attr("id", function (t, e) {
                return t._graphid
            }).attr("height", 1).attr("width", 1).attr("x", "0").attr("y", "0").append("image").attr("x", 0).attr("y", 0).attr("class", "objectImage").attr("xlink:href", function (t) {
                return t.image
            }).attr("preserveAspectRatio", "xMidYMid slice").attr("height", function (t, e) {
                return 2 * O(t, e)
            }).attr("width", function (t, e) {
                return 2 * O(t, e)
            }), n.append("circle").attr("class", "object").attr("cursor", "pointer").on("mouseover", function (t) {
                g || (u = !1, c.select(".title").text(t.name), c.select(".des").text(t.desc || "暂无简介"), c.select(".result-img a").style("background-image", "url('" + t.image + "')"), c.transition().duration(200).style("opacity", 1))
            }).on("mouseout", function (t) {
                g || (u = !0, c.transition().duration(200).style("opacity", 0))
            }).on("mousemove", function (t) {
                g || (1e3 < m ? (d3.event.pageX > m / 2 + 50 ? c.style("left", d3.event.pageX - 20 - 520 + "px") : d3.event.pageX < m / 2 - 50 ? c.style("left", d3.event.pageX + 10 + "px") : c.style("left", d3.event.pageX - 250 + "px"), d3.event.pageY > h / 2 ? c.style("top", d3.event.pageY - 150 + "px") : c.style("top", d3.event.pageY + 30 + "px")) : c.style("position", "fixed").style("width", "400px").style("height", "100px").style("left", "calc(50% - 200px)").style("top", "calc(50% - 50px)").style("display", "block"))
            }).on("click", function (t, e) {
                g || t._graphid.split("-")[0] == M.length - 1 && function (i, a) {
                    $("#search_input").val(""), x.selectAll("*").interrupt(), c.style("opacity", 0), setTimeout(function () {
                        c.style("opacity", 0)
                    }, 300), g = !0, u = !1;
                    var t = Math.random() * s / 2 - s / 4, e = p, n = t - i.angle;
                    s < n - e && (n -= o);
                    n - e < -s && (n += o);
                    $(".dummy").velocity({tween: [n, e]}, {
                        progress: function (t, e, n, r, a) {
                            p = a, F(!0)
                        }, complete: function (t) {
                            var e = L(i, a), n = 200;
                            e == n && (n += .1), W(110), $(".dummy").velocity({tween: [n, e]}, {
                                progress: function (t, e, n, r, a) {
                                    i.fixedDistance = a, F(!0)
                                }, complete: function (t) {
                                    var e = P(i, a, p), n = R(i, a, p), r = {
                                        category: i.category,
                                        name: i.name,
                                        image: i.image,
                                        entity: i.entity,
                                        x: e,
                                        y: n,
                                        relations: []
                                    };
                                    N(M, r), H(M.length - 1, function (t, e) {
                                        y = !0, e == M.length - 1 && (u = !0, t.entity ? q(t.entity, function (t, e) {
                                            y = !1, M[M.length - 1].relations = e, g = !1, I(), W(100)
                                        }) : (console.log("no entity"), console.log(t)))
                                    })
                                }
                            })
                        }
                    })
                }(t, e)
            }), n.append("text").attr("dy", "0.5em").attr("dx", "0").attr("class", "objectText").attr("visibility", "hidden"));
        r.append("tspan").style("fill", "white").text(function (t, e) {
            return t.predicate ? t.predicate + ": " : ""
        }), r.append("tspan").style("fill", "#f8af22").text(function (t, e) {
            var n = t.name;
            return 10 < n.length && (n = n.substring(0, 10) + "..."), n
        }), J(t)
    }

    function _(t) {
        return t.append("defs").attr("id", "imgdefs").append("pattern").attr("id", function (t) {
            return t.name + "pattern"
        }).attr("height", 1).attr("width", 1).attr("x", "0").attr("y", "0").append("image").attr("x", 0).attr("y", 0).attr("height", 50).attr("width", 50).attr("preserveAspectRatio", "xMidYMid slice").attr("xlink:href", function (t) {
            return t.image
        }), t.append("circle").attr("r", 25).attr("class", "subject").style("stroke", "#1d1d1d").style("stroke-width", 6).attr("fill", function (t) {
            return "url(#" + t.name + "pattern)"
        }).on("click", function (t, e) {
            U(e), H(e)
        })
    }

    function B(t) {
        return t.category ? e[t.category] : e[0]
    }

    function C(t) {
        return 5 < t && t < 20 ? "medium" : 20 <= t ? "big" : "small"
    }

    function L(t, e, n) {
        var r = t.distance - f, a = h / 3.8 + h / 40 * r;
        n || t.fixedDistance && (a = t.fixedDistance);
        return a < 50 && (a = e % 2 == 0 ? 50 : 40), a
    }

    function O(t, e) {
        var n = L(t, e), r = n / 30;
        return h / 10 < (r *= r) && (r = h / 10), $.each(t.sibs, function (t, e) {
            diff2 = e.distance - f, distance2 = h / 3.8 + h / 40 * diff2, distance2 > n / 2 && (r2 = Math.sqrt(-2 * distance2 * n * Math.cos(e.angle) + distance2 * distance2 + n * n), r2 = r2 * n / (n + distance2), r2 < r && 0 != e.angle && (r = r2))
        }), r < 5 && (r = 5), t.fixedDistance && (r = 25), r
    }

    function P(t, e, n) {
        return Math.cos(S(t, n)) * L(t, e)
    }

    function R(t, e, n) {
        return Math.sin(S(t, n)) * L(t, e)
    }

    function S(t, e) {
        var n = t.angle + e;
        return o <= n && (n -= o), n
    }

    function z() {
        var t = d3.select(".kg-graph .nodes-history").selectAll(".history-item").data(M);
        t.exit().remove();
        var e = t.enter().append("div").attr("class", "history-item");
        e.append("div").attr("class", "circle").style("background-color", function (t) {
            return B(t)
        }), e.append("div").attr("class", "text").text(function (t) {
            return t.name
        }), e.on("mouseover", function (t, e) {
            g || H(e)
        }).on("mouseout", function () {
            H(M.length - 1)
        }).on("click", function (t, e) {
            g || U(e)
        })
    }

    function J(t) {
        if (null != t && null != t.relations) {
            var e = d3.select(".kg-graph").select(".navigation-bar"), n = t.relations.sort(function (t, e) {
                return d3.ascending(t.distance, e.distance)
            }), r = 0, a = 0;
            0 < n.length && (r = n[0].distance, a = 1 < n.length ? n[n.length - 1].distance : r);
            var i = e.selectAll(".item").data(n);
            i.exit().remove(), i.enter().append("div").attr("class", "item").merge(i).style("background-color", function (t) {
                return B(t)
            }).style("left", function (t, e) {
                return 8 * (e % 8) + "px"
            }).style("bottom", function (t, e) {
                return l * (t.distance - r) / (a - r) - 4 + "px"
            }).on("mouseover", function (t) {
                g || (1e3 <= m ? c.style("left", d3.event.pageX - 40 - 530 + "px").style("top", d3.event.pageY - 80 + "px") : c.style("position", "fixed").style("width", "400px").style("height", "100px").style("left", "calc(50% - 200px)").style("top", "calc(50% - 50px)").style("display", "block"), c.select(".title").text(t.name), c.select(".des").text(t.desc || "暂无简介"), c.select(".result-img a").style("background-image", "url('" + t.image + "')"), c.transition().duration(200).style("opacity", 1))
            }).on("mouseout", function (t) {
                g || c.transition().duration(200).style("opacity", 0)
            }), e.on("mousemove", function (t) {
                var e = d3.event.y - 100;
                W(100 - (99 / l * e + 1))
            })
        }
    }

    function N(t, e) {
        t.push(e), x.select("#relation-container").selectAll(".relations").data(t).classed("current", !1).enter().append("g").attr("class", "relations current").attr("transform", function (t) {
            return "translate(" + t.x + "," + t.y + ")"
        }), I(), _(x.select("#node-container").selectAll(".node").data(t).enter().append("g").attr("class", "node").attr("transform", function (t) {
            return "translate(" + t.x + "," + t.y + ")"
        })), z()
    }

    function q(t, u) {
        $.get("/cndbpedia/kggraphData?entity=" + t, function (t) {
            var l = JSON.parse(t), r = [], c = l.relations.filter(function (t) {
                if (t.entity && t.entity != l.property.name) {
                    for (var e = !1, n = 0; n < M.length; n++) if (M[n].name == t.entity) {
                        e = !0;
                        break
                    }
                    if (!e) return t
                }
            }).map(function (t) {
                var e = t.pic ? t.pic : "/image/placeholder.jpg?v=2", n = t.click ? Math.log(t.click) : t.value.length;
                return -1 == r.indexOf(t.attr) && r.push(t.attr), {
                    predicate: t.attr,
                    desc: t.desc,
                    name: t.value,
                    image: e,
                    distance: n,
                    category: 0,
                    entity: t.entity
                }
            });
            $("#predicate-select").dropdown("clear"), $("#predicate-select").dropdown("setup menu", {
                values: [{
                    value: "-1",
                    text: "全部",
                    name: "全部"
                }].concat(r.map(function (t) {
                    return {value: t, text: t, name: t}
                }))
            }), $("#predicate-select").dropdown("setting", {
                onChange: function (e) {
                    -1 == e ? x.select("#relation-container").select(".current").selectAll(".relation").transition().duration(500).style("opacity", 1) : x.select("#relation-container").select(".current").selectAll(".relation").transition().duration(500).style("opacity", function (t) {
                        return t.predicate == e ? 1 : 0
                    })
                }
            }), null == l.property.related && (l.property.related = []);
            var a = l.property.related.filter(function (t) {
                if (t.o) {
                    var e;
                    if (null != (e = /<a\s+href=\"(.*?)\">(.*?)<\/a>/.exec(t.o))) {
                        t.entity = e[1], t.value = e[2];
                        for (var n = 0; n < c.length; n++) {
                            var r = c[n];
                            if (t.value === r.name || t.entity === r.name) return null
                        }
                        for (var a = 0; a < M.length; a++) {
                            var i = M[a];
                            if (t.entity === i.name) return null
                        }
                        if (t.entity === l.property.name) return null
                    }
                    return t
                }
            }).map(function (t) {
                var e = t.pic ? t.pic : "/image/placeholder2.jpg", n = t.click ? Math.log(t.click) : t.value.length;
                return {
                    predicate: t.attr,
                    desc: t.desc,
                    name: t.value,
                    image: e,
                    distance: n,
                    category: 1,
                    entity: t.entity,
                    context: t.context.replace(/<(?:.|\n)*?>/gm, "")
                }
            }), e = a.filter(function (t, e) {
                for (var n = -1, r = 0; r < a.length; r++) if (a[r].name == t.name) {
                    n = r;
                    break
                }
                return n == e
            });
            if (1 < (c = c.concat(e)).length) {
                var n = c.map(function (t) {
                    return t.distance
                }), i = d3.min(n), o = 99 / (d3.max(n) - i), s = 1 - o * i;
                c.forEach(function (t) {
                    t.distance = o * t.distance + s
                })
            } else 1 == c.length && (c[0].distance = 100);
            var d = l.property.pic ? l.property.pic : "/image/placeholder.jpg";
            return l.property.pic = d, u && u(l, c)
        })
    }

    function H(t, n) {
        x.selectAll("*").interrupt();
        var e = M[t], r = e.x, a = e.y;
        M.forEach(function (t) {
            t.x -= r, t.y -= a
        }), x.selectAll(".node").transition().duration(500).attr("transform", function (t) {
            return "translate(" + t.x + "," + t.y + ")"
        }), x.selectAll(".relations").transition().duration(500).attr("transform", function (t) {
            return "translate(" + t.x + "," + t.y + ")"
        }).on("end", function (t, e) {
            n && n(t, e)
        })
    }

    function U(t) {
        if (0 <= t && t < M.length - 1) {
            M.splice(t + 1, M.length - t);
            var e = M[t], n = [];
            e.relations.filter(function (t) {
                return void 0 !== t.predicate
            }).map(function (t) {
                var e = t.pic ? t.pic : "/image/placeholder.jpg?v=2";
                return -1 == n.indexOf(t.predicate) && n.push(t.predicate), {
                    predicate: t.attr,
                    desc: t.desc,
                    name: t.value,
                    image: e,
                    category: 0,
                    entity: t.entity
                }
            });
            $("#predicate-select").dropdown("clear"), $("#predicate-select").dropdown("setup menu", {
                values: [{
                    value: "-1",
                    text: "全部",
                    name: "全部"
                }].concat(n.map(function (t) {
                    return {value: t, text: t, name: t}
                }))
            });
            var r = x.select("#relation-container").selectAll(".relations").data(M);
            r.exit().remove(), r.classed("current", function (t, e) {
                return e == M.length - 1
            }), e.relations.forEach(function (t) {
                t.fixedDistance && delete t.fixedDistance
            }), x.select("#node-container").selectAll(".node").data(M).exit().remove(), J(e), z()
        }
    }

    function W(t) {
        if (t < 1 && (t = 1), 110 < t && (t = 110), t != f) {
            var e = {
                easing: "spring", progress: function (t, e, n, r, a) {
                    f = a
                }, complete: function (t) {
                }
            };
            $(".dummy2").velocity("stop"), $(".dummy2").velocity({tween: [t, f]}, e)
        }
    }

    function F(t) {
        (u || t) && (d3.selectAll(".current .objectAndText").attr("transform", function (t, e) {
            return "translate(" + P(t, e, p) + "," + R(t, e, p) + ")"
        }), d3.selectAll(".current .line").attr("x2", function (t, e) {
            return P(t, e, p)
        }).attr("y2", function (t, e) {
            return R(t, e, p)
        }), d3.selectAll(".current .objectImage").attr("height", function (t, e) {
            return 2 * O(t, e)
        }).attr("width", function (t, e) {
            return 2 * O(t, e)
        }), d3.selectAll(".current .object").attr("r", O).style("fill", function (t, e) {
            var n = C(O(t, e));
            return "medium" == n ? "black" : "big" == n ? "url(#" + t._graphid + ")" : B(t)
        }).style("stroke", function (t) {
            return B(t)
        }).style("stroke-width", function (t, e) {
            return 5 < O(t, e) ? 2 : 0
        }), d3.selectAll(".current .objectText").attr("visibility", function (t, e) {
            return "small" == C(O(t, e)) ? "hidden" : "visible"
        }).attr("transform", function (t, e) {
            var n = this.getBBox(), r = n.width, a = n.height, i = O(t, e), l = 0, c = 0, o = S(t, p);
            return 7 * s / 4 < o || o <= s / 4 ? (l = r / 2 + i + 15, c = 0) : 1 * s / 4 < o && o <= 3 * s / 4 ? (l = 0, c = i + 15 + a / 2) : 3 * s / 4 < o && o <= 5 * s / 4 ? (l = -(r / 2 + i + 15), c = 0) : (l = 0, c = -(i + 15 + a / 2)), "translate(" + l + "," + c + ")"
        })), t || (u && o <= (p += s / 3600) && (p -= o), y ? w.attr("visibility", "visible").attr("transform", function (t) {
            return "rotate(" + 3600 * p + ")"
        }) : w.attr("visibility", "hidden"), setTimeout(F, 10))
    }

    q(X, function (g, y) {
        t ? E(g, y) : $.get("/cndbpedia/kggraphConcepts?entity=" + X, function (t) {
            for (var e = JSON.parse(t), r = [], n = 0; n < e.count; n++) r.push(e.ret[n]);
            numSpirals = 3, start = 0, end = 2.5;
            var a = d3.min([m - 40, h - 40]) / 2, i = d3.range(start, end + .001, (end - start) / 1e3),
                l = d3.scaleLinear().domain([start, end]).range([10, a]),
                c = d3.radialLine().curve(d3.curveCardinal).angle(function (t) {
                    return numSpirals * Math.PI * t
                }).radius(l),
                o = x.select("#spiral-container").append("path").datum(i).attr("id", "spiral").attr("d", c).style("fill", "none").style("stroke", "gray"),
                s = o.node().getTotalLength();
            o.attr("stroke-dasharray", s + " " + s).attr("stroke-dashoffset", s).transition().duration(1e4).attr("stroke-dashoffset", 0), M.push({
                name: g.property.name,
                image: g.property.pic,
                x: 0,
                y: 0,
                relations: []
            });
            var d = x.select("#node-container").selectAll(".node").data(M).enter().append("g").attr("class", "node").attr("transform", function (t) {
                return "translate(" + t.x + "," + t.y + ")"
            }), u = _(d);
            u.style("stroke-width", 0);
            var p = s / r.length, f = x.select("#concept-container").selectAll(".concept").data(r).enter().append("g");
            f.attr("transform", function (t, e) {
                var n = o.node().getPointAtLength((e + .3) * p);
                return "translate(" + n.x + "," + n.y + ")"
            }), f.transition().duration(1e4).attrTween("opacity", function (t, n) {
                return function (t) {
                    if (t <= (n + .3) / r.length) return 0;
                    var e = 10 * (t - (n + .3) / r.length);
                    return 1 < e && (e = 1), e
                }
            }), f.append("circle").attr("r", 6).attr("fill", "darkgray"), f.append("text").text(function (t) {
                return t[0]
            }).attr("fill", "white").attr("dy", "0.5em").attr("transform", function (t, e) {
                var n = this.getBBox(), r = n.width, a = r / 2 + 15;
                return "translate(" + a + ",0)"
            }), setTimeout(function () {
                Y ? v.on("click", function () {
                    v.on("click", null), E(g, y)
                }) : E(g, y)
            }, 1e4)
        })
    })
});
