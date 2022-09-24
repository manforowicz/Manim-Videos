from manim import *
import random
from fractions import Fraction
random.seed(4334)

MY_RED = "#ff7575"
MY_BLUE = "#759cff"


def split(self):
    self.wait(5)
    self.next_section()


Scene.split = split


class Coin(VMobject):

    def __init__(self, type, radius=0.4, **kwargs):
        super().__init__(**kwargs)

        if type == 'H':
            self.add(
                Circle(color=WHITE, fill_color=BLUE_E,
                       fill_opacity=1, radius=radius, stroke_width=5*radius),
                Tex("H", font_size=135*radius)
            )
        else:
            self.add(
                Circle(color=WHITE, fill_color=RED_E,
                       fill_opacity=1, radius=radius, stroke_width=5*radius),
                Tex("T", font_size=135*radius)
            )

    @override_animation(Create)
    def _create_override(self):
        return AnimationGroup(
            FadeIn(self, scale=1.2, shift=DOWN * 0.2),
            self.animate.flip(),
        )


class CoinLine(VMobject):
    def __init__(self, sequence, **kwargs):
        super().__init__(**kwargs)
        self.add(*[Coin(t) for t in sequence]
                 ).arrange_in_grid(buff=0.15, cols=10)

    @override_animation(Create)
    def _Create_override(self, lag_ratio=0.5, run_time=1):
        return AnimationGroup(
            *[Create(c) for c in self],
            lag_ratio=lag_ratio,
            run_time=run_time
        )


class NChooseK(VGroup):
    def __init__(self, n, k, radius=0.2, **kwargs):
        super().__init__(**kwargs)

        tails = VGroup(
            *[Coin('T', radius=radius) for i in range(n)]
        ).arrange(buff=0.2*radius)
        if k == 0:
            self.add(tails)
            return

        chosen = list(reversed(range(k)))
        focus = 0
        while focus < k:
            if chosen[focus] >= n-focus:
                focus += 1
            else:
                if focus == 0:
                    new = tails.copy()
                    for pos in chosen:
                        new[pos] = Coin('H', radius=radius).move_to(new[pos])
                    self.add(new)

                chosen[focus] += 1
                for i in range(0, focus):
                    chosen[i] = chosen[focus] + focus-i
                focus = 0


class MoneyTree(VGroup):
    def __init__(self, ax, multiplicative=True, **kwargs):
        super().__init__(**kwargs)
        layers = ax.x_range[1] + 1
        self.ax = ax
        self.multiplicative = multiplicative

        for layer in range(layers):
            for losses in range(layer+1):
                gains = layer-losses
                me = self.get_point(gains, losses)
                if losses != 0:
                    other = self.get_point(gains, losses-1)
                    self.link(other, me, MY_RED)
                if gains != 0:
                    other = self.get_point(gains-1, losses)
                    self.link(other, me, MY_BLUE)

                self.add(
                    Circle(
                        radius=0.05,
                        stroke_width=0,
                        fill_opacity=1,
                        fill_color=WHITE
                    ).move_to(me)
                )

    def get_point(self, gains, losses):
        if self.multiplicative:
            y = 100 * 1.8 ** gains * 0.5 ** losses
        else:
            y = 100 + 40 * gains - 25 * losses
        return self.ax.coords_to_point(gains+losses, y)

    def link(self, a, b, color):
        self.add(Arrow(
            a,
            b,
            max_tip_length_to_length_ratio=0.05,
            stroke_width=2,
            buff=0.05,
            color=color
        ))

    def paths_to(self, n, k):
        points = []
        paths = []

        def func(g, l):
            points.append(self.get_point(g, l))
            if g + l >= n:
                path = VGroup()
                for i in range(len(points)-1):
                    path += Line(points[i], points[i+1],
                                 color=YELLOW, stroke_width=8)

                paths.append(path)
                points.pop()
                return

            if g < k:
                func(g+1, l)
            if l < n-k:
                func(g, l+1)
            points.pop()

        func(0, 0)
        return paths


class CoinSim(VGroup):
    def __init__(self, sequence, **kwargs):
        super().__init__(**kwargs)
        wealth = 100
        line = [MathTex(r"\$" + str(wealth))]
        for l in sequence:
            line.append(Coin(l).next_to(line[-1], LEFT))
            factor = 0.5
            if l == 'H':
                factor = 1.8
            line.append(MathTex(r"\times" + str(factor)).next_to(line[-2]))
            wealth *= factor

            line.append(
                Tex(r"\$" + "{:.0f}".format(wealth)).next_to(line[-2]).shift(RIGHT))
        self.line = VGroup(*line)
        self.add(self.line)

    def animate(self, scene):
        scene.play(Write(self.line[0]))
        scene.split()
        i = 0
        for i in range(0, len(self.line)-1, 3):
            scene.play(Create(self.line[i+1]), FadeIn(self.line[i+2]))
            scene.split()
            scene.play(ReplacementTransform(
                VGroup(self.line[i], self.line[i+2]), self.line[i+3]))

        scene.split()
        scene.play(*[FadeOut(self.line[i])
                   for i in range(1, len(self.line), 3)], FadeOut(self.line[-1]))


# SCENES


class Intro(Scene):

    def construct(self):
        game = Tex(
            "The ",  "``Just One More''", " Paradox",
            stroke_color=BLUE
        ).scale(2)
        game[1].set_color(LIGHT_PINK)
        coins = CoinLine("HTHHTHTTHTHHTHTHHHTHTHTHHTHTHHTHHTHTHTTHTHHTH")
        for c in coins:
            c.move_to(
                RIGHT*np.random.uniform(-6.5, 6.5) +
                UP*np.random.choice(np.random.uniform([-3.5, 2], [-2, 3.5]))
            )
        self.play(
            Create(coins, run_time=8, lag_ratio=0.1),
            AnimationGroup(
                FadeIn(game[0]),
                Write(game[1]),
                SpinInFromNothing(game[2]),
                lag_ratio=1,
                run_time=5
            ),
        )
        self.play(FadeOut(game), FadeOut(coins))


class Rules(Scene):
    def construct(self):

        # Play table
        self.play(Write(
            VGroup(
                Line(UP*3.5, UP*1.5),
                Line(UP*2.5 + LEFT*5, UP*2.5 + RIGHT*5),
            )
        ))

        self.split()

        # Play coins
        self.play(Create(Coin("H").move_to(UP*3.2 + LEFT*1.5)))

        self.play(Write(
            MathTex(r"\uparrow +80\%", color=MY_BLUE)
            .move_to(UP*2 + LEFT, RIGHT)
        ))
        self.split()

        self.play(Create(Coin("T").move_to(UP*3.2 + RIGHT*1.5)))

        self.play(Write(
            MathTex(r"\downarrow -50\%", color=MY_RED)
            .move_to(UP*2 + RIGHT, LEFT)
        ))

        self.split()

        # Play examples
        coins = CoinSim("HHTT").shift(DOWN+LEFT)
        coins.animate(self)
        self.split()

        # Write Probability Equation
        eq = MathTex(
            r"\frac{1}{2} \times 0.8",
            r"+", r"\frac{1}{2} \times -0.5",
            r"="
        ).shift(RIGHT*0.45 + UP*0.6)

        eq[0].set_color(MY_BLUE)
        eq[2].set_color(MY_RED)

        for part in eq:
            self.play(Write(part))
            self.wait(1.5)

        eq2 = Tex(
            r"= 0.15 ",
            r"= 15\% ",
            r"average gain per coin toss"
        ).shift(DOWN)

        self.play(Transform(eq.copy(), eq2[0]))
        self.split()
        self.play(Transform(eq2[0].copy(), eq2[1]))
        self.split()
        self.play(FadeIn(eq2[2]))

        self.split()
        self.play(SpinInFromNothing(
            Tex("Sounds great!", color=YELLOW).scale(2).shift(3*DOWN)))
        self.split()


class Simulation(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0, 55, 50],
            y_range=[0, 105000, 50000],
            axis_config={"include_numbers": True},
            x_length=11
        ).move_to(RIGHT*0.5)
        x_label = ax.get_x_axis_label(
            Tex("tosses"), edge=DOWN, direction=DOWN)
        y_label = ax.get_y_axis_label(
            Tex(r"\$"), direction=UL)
        start = Tex("Start: \\$100", font_size=36).next_to(ax.c2p(0, 100), LEFT)

        self.play(Write(ax), Write(x_label), Write(y_label), Write(start))

        self.split()

        coins = CoinLine(np.random.choice(['H', 'T'], 50)).move_to(RIGHT)
        self.play(Create(coins, lag_ratio=0.1), run_time=5)
        self.play(FadeOut(coins))

        print("SIMULATING TOSSES")
        sim = [0] * 51
        for i in range(500000):
            wealth = 100
            for j in range(len(sim)):
                sim[j] += wealth
                if bool(random.getrandbits(1)):
                    wealth *= 1.8
                else:
                    wealth *= 0.5
        for i in range(len(sim)):
            sim[i] /= 500000

        line_graph = ax.plot_line_graph(
            x_values=range(51),
            y_values=sim,
            line_color=PURPLE_A,
            vertex_dot_radius=0.06,
            stroke_width=8
        )

        self.play(FadeIn(line_graph))
        self.split()
        value = ax.get_horizontal_line(ax.c2p(50, sim[50], 0), color=PURPLE_A)
        good = Tex(r"final average \$" +
                   "{:.0f}".format(sim[50]), color=PURPLE_A).move_to(UP*2.8+RIGHT*2)
        self.play(Write(value), FadeIn(good))
        self.split()

        sim_median = [0]*51
        for i in range(51):
            sim_median[i] = 100 * (1.8*0.5) ** (i*0.5)

        median_graph = ax.plot_line_graph(
            x_values=range(51),
            y_values=sim_median,
            line_color=ORANGE,
            vertex_dot_radius=0.06,
            stroke_width=8
        )
        self.play(FadeIn(median_graph))

        measly = Tex(r"final median \$7.2", color=ORANGE).move_to(
            2.2*DOWN+4.9*RIGHT)
        self.play(FadeIn(measly))
        self.split()
        what = Tex(r"HOW!?", color=LIGHT_PINK).scale(2)
        self.play(GrowFromCenter(what))
        self.split()


class TwoTosses(Scene):
    def construct(self):
        coins = VGroup(
            CoinLine("HH"),
            CoinLine("HT"),
            CoinLine("TH"),
            CoinLine("TT"),
        ).arrange(direction=DOWN, buff=0.5).move_to(UP*1.3 + LEFT*5.5)

        eqs = [
            MathTex(r"\$100 \times 1.8 \times 1.8").next_to(coins[0]),
            MathTex(r"\$100 \times 1.8 \times 0.5").next_to(coins[1]),
            MathTex(r"\$100 \times 0.5 \times 1.8").next_to(coins[2]),
            MathTex(r"\$100 \times 0.5 \times 0.5").next_to(coins[3]),
        ]
        ans = VGroup(
            MathTex(r"\$324").next_to(coins[0]),
            MathTex(r"\$90").next_to(coins[1]),
            MathTex(r"\$90").next_to(coins[2]),
            MathTex(r"\$25").next_to(coins[3])
        )

        for i in range(4):
            self.play(Create(coins[i]))
            self.play(Write(eqs[i]))
            self.wait(1.5)
            self.play(Transform(eqs[i], ans[i]))
            self.split()

        avg = VGroup(
            MathTex(r"\frac{324+90+90+25}{4}"),
            MathTex(r"\$132.25")
        ).move_to(3*UP + 3*RIGHT)

        avg_label = MathTex("avg=").next_to(avg[1], LEFT).shift(DOWN*0.1)

        self.play(TransformFromCopy(ans, avg[0]))
        self.wait(2)
        self.play(Transform(avg[0], avg[1]), FadeIn(avg_label))
        self.split()
        self.play(FadeIn(MathTex(r"=100 \times 1.15^2").next_to(avg[1])))
        self.split()

        pointer = VGroup(
            Vector(LEFT),
            MathTex("median")
        ).arrange().next_to(ans[0]).shift(UP*2)

        self.play(pointer.animate.shift(DOWN*6))
        self.play(pointer.animate.next_to(ans))
        pointer_val = MathTex(r"=\$90").next_to(pointer)
        self.play(FadeIn(pointer_val))

        self.split()

        brace = VGroup(
            Brace(VGroup(ans[1], ans[2]), direction=RIGHT),
        )
        brace.add(MathTex("mode").next_to(brace))
        self.play(Transform(pointer, brace),
                  pointer_val.animate.next_to(brace))

        self.split()

        self.play(CyclicReplace(coins[1][0], coins[1][1]))
        self.play(CyclicReplace(coins[2][0], coins[2][1]))

        self.split()
        rect = SurroundingRectangle(VGroup(
            coins[1][1], coins[2][1], ans[1], ans[2]
        ))
        rect_label = Tex(r"half H, half T", color=YELLOW).next_to(
            rect, UR).shift(LEFT * 0.1 + DOWN*0.5)
        self.play(Write(rect), Write(rect_label))
        self.split()


class HalfNHalf(Scene):
    def construct(self):

        chart = BarChart(
            values=[1, 8, 28, 56, 70, 56, 28, 8, 1],
            bar_names=["0H", "1H", "2H", "3H", "4H", "5H", "6H", "7H", "8H"],
            y_range=[0, 71, 70],
            y_length=6,
            x_length=10,
            x_axis_config={"font_size": 42},
            bar_colors=reversed([
                "#bc5090",
                "#ff6361",
                "#ffa600"])
        )
        chart_labels = chart.get_bar_labels(font_size=48)

        all = NChooseK(8, 0, radius=0.4)
        self.play(AnimationGroup(
            *[Create(c) for c in all[0]],
            lag_ratio=0.2,
            run_time=2
        ))
        self.split()
        self.play(FadeIn(chart[1:]))
        self.split()
        self.play(FadeTransform(all, chart[0][0]), FadeIn(chart_labels[0]))
        self.split()

        for i in range(1, 9):
            all = NChooseK(8, i, radius=0.15).arrange_in_grid(
                flow_order="dr", buff=(0.2, 0.1), rows=14)
            self.play(AnimationGroup(
                *[FadeIn(line) for line in all],
                lag_ratio=0.5,
            ))
            if i < 5:
                self.wait(1)
            self.play(FadeTransform(all, chart[0][i]), FadeIn(chart_labels[i]))

        self.split()
        self.play(Indicate(chart_labels[4]), Indicate(chart[0][4]), run_time=2)
        self.split()


class DecisionTree(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 1001, 100],
            tips=False,
            axis_config={"include_numbers": True},
        )
        x_label = ax.get_x_axis_label(
            Tex("tosses"), edge=DOWN, direction=DOWN).shift(DOWN*0.1)
        y_label = ax.get_y_axis_label(
            Tex(r"\$"), direction=UL)
        tree = MoneyTree(ax)
        avg = ax.plot(lambda x: 100*1.15**x)
        avg_label = Tex("avg").next_to(avg.get_point_from_function(6), RIGHT)

        log_ax = Axes(
            x_range=[0, 6, 1],
            y_range=[0.1, 3.3, 1],
            tips=False,
            axis_config={"include_numbers": True},
            y_axis_config={"scaling": LogBase()}
        )
        log_tree = MoneyTree(log_ax)
        log_avg = log_ax.plot(lambda x: 100*1.15**x)
        log_median = log_ax.plot(lambda x: 100 * 1.8 **
                                 (0.5*x) * 0.5**(0.5*x), color=YELLOW)
        log_avg_label = Tex("avg").next_to(
            log_avg.get_point_from_function(6), DOWN)

        small_ax = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 201, 100],
            tips=False,
            axis_config={"include_numbers": True},
        )
        small_tree = MoneyTree(small_ax)
        small_avg = small_ax.plot(lambda x: 100*1.15**x)
        small_median = small_ax.plot(
            lambda x: 100 * 1.8**(0.5*x) * 0.5**(0.5*x), color=YELLOW)
        small_avg_label = Tex("avg").next_to(
            small_avg.get_point_from_function(6), DOWN)

        small_mode_label = Tex("mode", color=YELLOW).next_to(
            small_median.get_point_from_function(6), DOWN)

        minus = MathTex(r"\times 0.5 \rightarrow \$50",
                        color=MY_RED).next_to(tree[4])
        plus = MathTex(r"\times 1.8 \rightarrow \$180",
                       color=MY_BLUE).next_to(tree[2])

        self.play(Write(ax), Write(y_label), Write(x_label))
        self.split()
        self.play(Create(tree[0]), Flash(tree[0]), Indicate(tree[0]))
        self.split()
        self.play(Create(tree[1:5], run_time=3), AnimationGroup(
            FadeIn(plus),
            FadeIn(minus),
            lag_ratio=1.5,
            run_time=2
        ))
        already = Tex(r"\$90").next_to(tree[9])
        already_line = ax.get_horizontal_line(ax.c2p(2, 90))

        self.split()
        self.play(FadeOut(plus), FadeOut(minus))
        self.play(Create(tree[5:10]), run_time=4)
        self.play(FadeIn(already), Write(already_line))
        self.split()
        self.play(FadeOut(already), FadeOut(already_line))
        self.split()
        self.play(Create(tree[10:22]), run_time=4)
        self.play(Create(tree[22:]), run_time=4)
        self.split()
        self.play(Create(avg), FadeIn(avg_label))

        self.split()
        self.play(
            ReplacementTransform(avg_label, log_avg_label),
            FadeOut(ax),
            FadeIn(log_ax),
            ReplacementTransform(tree, log_tree),
            ReplacementTransform(avg, log_avg),
            run_time=5
        )
        self.split()

        add = Tex("View: Logarithmic").add_background_rectangle()
        self.play(FadeIn(add))
        self.split()
        self.play(FadeOut(add))
        self.split()

        point = Dot(log_tree.get_point(3, 3))
        point_label = Tex("3H, 3T", color=YELLOW).next_to(point, DOWN)
        self.play(FadeIn(point_label), Flash(point))
        self.split()

        paths = log_tree.paths_to(6, 3)

        self.play(Create(paths[0]), run_time=2)
        self.play(FadeOut(point_label), FadeOut(point))

        i = -1
        center_label = Integer(1, color=YELLOW).next_to(point)
        center_label.add_updater(
            lambda center_label: center_label.set_value(i+2))
        self.play(FadeIn(center_label))

        self.split()

        for i in range(len(paths)-1):
            self.remove(paths[i])
            self.add(paths[i+1])
            center_label.update()
            self.wait(0.5)
        self.remove(paths[i+1])
        center_label.clear_updaters()
        self.split()

        labels = VGroup()
        for heads in range(4, 7):

            i = -1
            label = Integer(1).next_to(log_tree.get_point(heads, 6-heads))
            labels += label
            label.add_updater(lambda label: label.set_value(i+2))
            paths = log_tree.paths_to(6, heads)
            self.add(label)
            self.play(Create(paths[0]))
            self.wait(0.5)
            for i in range(len(paths)-1):
                self.remove(paths[i])
                self.add(paths[i+1])
                label.update()
                self.wait(0.5)
            self.remove(paths[i+1])
            label.clear_updaters()
        self.split()

        flip_labels = VGroup(
            Integer(15).next_to(log_tree.get_point(2, 4)),
            Integer(6).next_to(log_tree.get_point(1, 5)),
            Integer(1).next_to(log_tree.get_point(0, 6))
        )

        self.play(TransformFromCopy(
            labels, flip_labels, path_arc=-3), run_time=2)
        self.split()

        self.play(Indicate(center_label), Flash(point))
        self.split()
        self.play(ReplacementTransform(center_label, point_label))

        median_label = Tex("median", color=YELLOW).next_to(point, DOWN)
        mode_label = Tex("mode", color=YELLOW).next_to(point, DOWN)
        self.play(Create(log_median), run_time=5)
        self.split()
        self.play(ReplacementTransform(point_label, median_label))
        self.wait(1)
        self.play(ReplacementTransform(median_label, mode_label))
        self.split()

        self.play(FadeOut(labels), FadeOut(flip_labels))

        self.split()
        ref = ax.get_lines_to_point(small_ax.c2p(6, 100), color=GRAY)
        self.play(
            FadeOut(log_ax),
            FadeIn(small_ax),
            FadeIn(ref[0]),
            ReplacementTransform(mode_label, small_mode_label),
            ReplacementTransform(log_avg_label, small_avg_label),
            ReplacementTransform(log_tree, small_tree),
            ReplacementTransform(log_avg, small_avg),
            ReplacementTransform(log_median, small_median),
            run_time=5
        )

        self.split()

        def paradox(tree, ax, gains, losses):

            home = tree.get_point(gains, losses)
            loss = tree.get_point(gains, losses+1)
            gain = tree.get_point(gains+1, losses)

            gain_val = ax.p2c(gain)[1] - ax.p2c(home)[1]
            loss_val = ax.p2c(home)[1] - ax.p2c(loss)[1]

            loss_label = MathTex(
                r"- \$" + "{:.0f}".format(loss_val),
                color=MY_RED
            ).next_to(loss, UP).shift(UP*0.006*loss_val)

            gain_label = MathTex(
                r"+ \$" + "{:.0f}".format(gain_val),
                color=MY_BLUE
            ).next_to(gain, DOWN).shift(DOWN*0.006*gain_val)

            home = Dot(home, color=YELLOW)

            loss = Line(home, loss,
                        color=RED, stroke_width=10)
            gain = Line(home, gain,
                        color=BLUE, stroke_width=10)
            
            

            self.play(Create(home), Flash(home),
                      ShowPassingFlash(gain, time_width=2),
                      FadeIn(gain_label), run_time=2)
            self.play(ShowPassingFlash(loss, time_width=2),
                      FadeIn(loss_label), run_time=2)
            self.split()
            self.play(FadeOut(home))
            return VGroup(gain_label, loss_label)

        paradox_labels = VGroup()
        paradox_labels += paradox(small_tree, small_ax, 0, 0)
        paradox_labels += paradox(small_tree, small_ax, 2, 2)
        paradox_labels += paradox(small_tree, small_ax, 0, 1)
        self.split()
        self.play(FadeOut(paradox_labels))
        self.split()

        additive_tree = MoneyTree(small_ax, multiplicative=False)
        additive_median = small_ax.plot(
            lambda x: 100 + 20*x - 12.5*x, color=YELLOW)

        additive_avg = small_ax.plot(
            lambda x: 100 + 20*x - 12.5*x)

        additive_avg_label = Tex("avg").next_to(
            additive_avg.get_point_from_function(6), DOWN)

        additive_mode_label = Tex("mode", color=YELLOW).next_to(
            additive_median.get_point_from_function(6), DOWN)

        mult = Tex("Game: Multiplicative").add_background_rectangle()
        self.play(FadeIn(mult))
        self.wait()
        self.play(FadeOut(mult))
        self.split()

        all_wealth = Tex("Bet entire wealth").move_to(UP*0.4)
        cross = Cross().scale(0.2).stretch_to_fit_width(3).move_to(all_wealth)
        fixed = Tex(r"Always bet \$50", color=GREEN).next_to(all_wealth, DOWN)
        back = BackgroundRectangle(VGroup(all_wealth, cross, fixed))

        self.play(FadeIn(back), FadeIn(all_wealth))
        self.split()
        self.play(Write(cross))
        self.play(GrowFromCenter(fixed))
        self.split()

        self.play(FadeOut(all_wealth), FadeOut(cross), FadeOut(fixed), FadeOut(back))
        self.split()

        self.play(
            Transform(small_tree, additive_tree),
            Transform(small_median, additive_median),
            Transform(small_avg, additive_avg),
            Transform(small_avg_label, additive_avg_label),
            Transform(small_mode_label, additive_mode_label),
            run_time=5)
        self.split()

        add = Tex("Game: Additive").add_background_rectangle()
        self.play(FadeIn(add))
        self.split()
        self.play(FadeOut(add))
        self.split()

        paradox_labels = VGroup()
        paradox_labels += paradox(additive_tree, small_ax, 1, 1)
        paradox_labels += paradox(additive_tree, small_ax, 0, 1)
        self.split()
        self.play(FadeOut(paradox_labels))
        self.split()

        low = Circle(stroke_color=RED, radius=1.5, fill_opacity=0.1,
                     color=YELLOW).stretch_to_fit_width(6).move_to(RIGHT*3.5 + DOWN * 2.5)
        self.play(Create(low))
        self.split()

        high = Arrow(UP*2 + RIGHT*2, UP*4 + RIGHT*2.5, stroke_width=20)
        high_label = Tex("?").scale(2.5).next_to(high, DOWN)
        self.play(Create(high))
        self.play(FadeIn(high_label))
        self.split()


class WealthFraction(Scene):
    def construct(self):
        # Cross out old
        a = Tex("Bet entire wealth").move_to(UP*3.5)
        b = Cross().scale(0.2).stretch_to_fit_width(3).move_to(a)
        c = Tex(r"Always bet \$50").next_to(a, DOWN)
        d = Cross().scale(0.2).stretch_to_fit_width(3).move_to(c)
        e = Tex(r"Always bet $\frac{1}{5}$ of our wealth", color=GREEN).next_to(
            c, DOWN)
        f = Tex(r"Always bet $\frac{1}{10}$ of our wealth", color=GREEN).next_to(
            c, DOWN)
        g = Tex(r"Always bet ??? of our wealth", color=GREEN).next_to(
            c, DOWN)
        old = VGroup(a, c)
        cross = VGroup(b, d)
        self.play(FadeIn(old))
        self.play(Write(cross))
        self.play(GrowFromCenter(e))
        self.split()
        self.play(FadeTransform(e, f))
        self.split()

        # Play table
        self.play(Write(
            VGroup(
                Line(DOWN*0.8, UP*1.5),
                Line(UP*0.5 + LEFT*3.5, UP*0.5 + RIGHT*3.5),
            )
        ))
        self.split()

        # Play coins
        self.play(Create(Coin("H").move_to(UP*1.2 + LEFT*1.5)))
        heads_eq = MathTex(
            r"\times", r"(1 +", r"\frac{1}{10} \times", "0.8)", color=MY_BLUE).move_to(LEFT*0.5+0.3*DOWN, RIGHT)
        heads_eq[2].set_color(GREEN)

        self.play(Write(heads_eq[0]))
        self.play(Write(heads_eq[1]), Write(heads_eq[3]))
        self.split()
        self.play(FadeIn(heads_eq[2]))
        self.split()

        self.play(Create(Coin("T").move_to(UP*1.2 + RIGHT*1.5)))
        tails_eq = MathTex(
            r"\times", r"(1 -", r"\frac{1}{10} \times", "0.5)", color=MY_RED).move_to(RIGHT*0.5+0.3*DOWN, LEFT)
        tails_eq[2].set_color(GREEN)

        self.play(Write(tails_eq[0]))
        self.play(Write(tails_eq[1]), Write(tails_eq[3]))
        self.split()
        self.play(FadeIn(tails_eq[2]))
        self.split()

        # Play repeated multiplication
        a = VGroup(
            heads_eq.copy()[1:],
            heads_eq.copy()[1:],
            MathTex(r"\times"),
            tails_eq.copy()[1:],
            tails_eq.copy()[1:],
        ).arrange().shift(3*DOWN)

        b = MathTex(r"(1 +", r"\frac{1}{10}", r"\times 0.8)", "^ 2", r"\times",
                    r"(1 -", r"\frac{1}{10}", r"\times 0.5)", "^ 2"
                    ).shift(3*DOWN)
        b[0:3].set_color(MY_BLUE)
        b[1].set_color(GREEN)
        b[3].set_color(YELLOW)
        b[5:8].set_color(MY_RED)
        b[6].set_color(GREEN)
        b[8].set_color(YELLOW)

        c = MathTex(r"(1 +", r"\frac{1}{10}", r"\times 0.8)", "^ 1", r"\times",
                    r"(1 -", r"\frac{1}{10}", r"\times 0.5)", "^ 1"
                    ).shift(3*DOWN)
        c[0:3].set_color(MY_BLUE)
        c[1].set_color(GREEN)
        c[3].set_color(YELLOW)
        c[5:8].set_color(MY_RED)
        c[6].set_color(GREEN)
        c[8].set_color(YELLOW)

        d = MathTex(r"(1 +", r"\frac{1}{10}", r"\times 0.8)", "^ {0.5}", r"\times",
                    r"(1 -", r"\frac{1}{10}", r"\times 0.5)", "^ {0.5}"
                    ).shift(3*DOWN)
        d[0:3].set_color(MY_BLUE)
        d[1].set_color(GREEN)
        d[3].set_color(YELLOW)
        d[5:8].set_color(MY_RED)
        d[6].set_color(GREEN)
        d[8].set_color(YELLOW)

        a_coins = [
            Coin('H').next_to(a[0], UP),
            Coin('H').next_to(a[1], UP),
            VMobject(),
            Coin('T').next_to(a[3], UP),
            Coin('T').next_to(a[4], UP),
        ]

        half_blocker = Rectangle(
            color=BLACK, fill_opacity=1, height=1.2, width=3.7).move_to(1.7*DOWN)

        r = MathTex("r=").next_to(d, LEFT)
        result = MathTex("=1.013").next_to(d, RIGHT)

        self.play(
            AnimationGroup(
                *[FadeIn(elem) for elem in a],
                lag_ratio=2,
                run_time=8
            ),
            AnimationGroup(
                *[Create(elem) for elem in a_coins],
                lag_ratio=2,
                run_time=8
            )
        )

        self.split()
        self.play(FadeTransform(a[0:2], b[0:4]))
        self.split()
        self.play(FadeTransform(a[3:5], b[5:9]))
        self.play(FadeTransform(b[0:4], c[0:4]), FadeTransform(
            b[5:9], c[5:9]), FadeOut(a_coins[0]), FadeOut(a_coins[4]))
        self.split()
        self.play(FadeTransform(c[0:4], d[0:4]), FadeTransform(
            c[5:9], d[5:9]), GrowFromCenter(half_blocker))
        self.split()
        self.play(Write(r))
        self.split()
        self.play(Write(result))
        self.split()
        self.play(FadeOut(result))
        self.play(Indicate(f))
        self.play(ReplacementTransform(f, e))
        self.wait(2)
        self.play(ReplacementTransform(e, g))
        self.split()

        # Replace all with variables

        legend = Rectangle(height=3.2, width=3.2).move_to(UP*2.3 + RIGHT*5)
        vars = VGroup(
            MathTex("f = 0.1", color=GREEN),
            MathTex("b = 0.8", color=MY_BLUE),
            MathTex("a = 0.5", color=MY_RED),
            MathTex("p = 0.5", color=YELLOW),
            MathTex("q = 0.5", color=YELLOW)
        ).arrange(DOWN, buff=0.15).move_to(UP*2.2 + RIGHT*5)

        self.play(FadeIn(legend))
        for var in vars:
            self.play(Write(var))
            self.split()

        eq = MathTex(r"(1 +", r"f", r"b)", "^ p", r"\times",
                     r"(1 -", r"f", r"a)", "^ q"
                     ).shift(3*DOWN)
        eq[0:3].set_color(MY_BLUE)
        eq[1].set_color(GREEN)
        eq[3].set_color(YELLOW)
        eq[5:8].set_color(MY_RED)
        eq[6].set_color(GREEN)
        eq[8].set_color(YELLOW)

        self.play(FadeTransform(d[0:4], eq[0:4]), FadeTransform(
            d[5:9], eq[5:9]), r.animate.next_to(eq, LEFT))
        self.split()


class KellyGraph(Scene):
    def construct(self):
        # Set up Graph
        ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[0.9, 1.1, 0.1],
            x_length=10,
            tips=False,
            axis_config={"include_numbers": True,
                         "exclude_origin_tick": False},
        ).shift(LEFT*0.5)
        ax[0].set_color(GREEN)
        x_label = ax.get_x_axis_label(
            MathTex("f"), edge=DOWN, direction=DR).set_color(GREEN).shift(RIGHT*0.5)
        y_label = ax.get_y_axis_label(
            MathTex(r"r"), edge=LEFT, direction=UL).shift(UP*0.5)
        zero_label = MathTex("0", font_size=36, color=GREEN).next_to(
            ax.c2p(0, 0.9), DOWN)
        ref = ax.get_lines_to_point(ax.c2p(1, 1))

        graph = ax.plot(lambda x: (1+x*0.8)**0.5 * (1-x*0.5)**0.5)
        gain_graph = ax.plot(lambda x: (1+x*0.8)**0.5 * (1-x*0.5) **
                             0.5, x_range=[0, 0.75], stroke_width=20, color="#369D00")

        eq = MathTex("r=", r"(1 +", r"f", r"b)", "^ p", r"\times",
                     r"(1 -", r"f", r"a)", "^ q"
                     ).move_to(UP*3)
        eq[1:4].set_color(MY_BLUE)
        eq[2].set_color(GREEN)
        eq[4].set_color(YELLOW)
        eq[6:9].set_color(MY_RED)
        eq[7].set_color(GREEN)
        eq[9].set_color(YELLOW)

        self.play(FadeIn(eq))
        self.split()
        self.play(Write(VGroup(ax, x_label, y_label, ref, zero_label)))
        self.wait()
        self.play(Write(graph), run_time=2)
        self.split()

        # Explain x and y axis

        self.play(Indicate(ax[0]), Indicate(x_label))
        self.split()
        self.play(Indicate(ax[1]), Indicate(y_label))
        self.split()

        tracker = ValueTracker(0)

        def point_updater(p):
            pos = graph.point_from_proportion(tracker.get_value())
            coords = ax.p2c(pos)
            p[0].move_to(pos)
            p[1].next_to(pos, UR).shift(UP*0.4 + LEFT*0.2)
            p[2].set_value(coords[0]).move_to(p[1]).shift(LEFT*0.6)
            p[3].set_value(coords[1]).move_to(p[1]).shift(RIGHT*0.6)

        point = VGroup(
            Dot(color=ORANGE),
            Tex(r"(\;\;\;\;\;\;\;,\;\;\;\;\;\;\;)"),
            DecimalNumber(font_size=36, num_decimal_places=3, color=GREEN),
            DecimalNumber(font_size=36, num_decimal_places=3)
        ).add_updater(point_updater)

        def tangent_updater(t):
            t.become(TangentLine(graph, tracker.get_value(), 3, color=ORANGE))

        tangent = TangentLine(graph, 1).add_updater(tangent_updater)

        self.play(FadeIn(point))
        self.play(Indicate(point[3]))
        self.split()

        self.play(tracker.animate.set_value(1), run_time=2)
        self.split()
        self.play(Indicate(point[3]))
        self.split()

        self.play(Create(gain_graph), run_time=3)
        self.play(Indicate(gain_graph))
        self.split()

        self.play(Write(tangent), FadeOut(gain_graph))
        self.play(tracker.animate.set_value(0.25), run_time=2)
        self.wait()
        self.play(tracker.animate.set_value(0.5), run_time=3)
        self.wait()
        self.play(tracker.animate.set_value(0.2), run_time=3)
        self.wait()
        self.split()

        cover = Rectangle(color=BLACK, fill_opacity=1, width=14, height=10)
        self.add(cover)
        self.bring_to_front(eq)
        self.split()

        kelly = MathTex(
            r"f^*", "=", r"{p", r"\over", r"a}", "-", r"{q", r"\over", r"b}", r"=0.375")
        kelly.set_color_by_tex("f^*", GREEN)
        kelly.set_color_by_tex("p", YELLOW)
        kelly.set_color_by_tex("a", MY_RED)
        kelly.set_color_by_tex("q", YELLOW)
        kelly.set_color_by_tex("b", MY_BLUE)
        kelly.set_color_by_tex("=0.375", GREEN)

        steps = VGroup(
            MathTex(r"\ln{(r)} = \ln{((1+fb)^p)} + \ln{((1-fa)^q)}"),
            MathTex(r"\ln{(r)} = p\ln{(1+fb)} + q\ln{(1-fa)}"),
            MathTex(
                r"\frac{1}{r} \times \frac{dr}{df} = \frac{pb}{1+fb}+ \frac{-qa}{1-fa}"),
            MathTex(r"0= r(\frac{pb}{1+f^*b}+ \frac{-qa}{1-f^*a})"),
            kelly[:-1]
        ).arrange(DOWN).next_to(eq, DOWN)

        self.play(FadeTransform(eq.copy(), steps[0]))

        for i in range(1, len(steps)):
            self.wait(2)
            self.play(FadeTransform(steps[i-1].copy(), steps[i]))

        self.split()
        self.play(FadeOut(steps[:-1]))
        self.play(kelly[:-1].animate.next_to(eq, DOWN).shift(RIGHT*4))
        self.play(FadeOut(cover))
        self.split()

        self.play(Indicate(kelly[:-1]))
        self.split()
        self.play(Write(kelly[-1].next_to(kelly[:-1]).shift(UP*0.05)))
        self.wait()

        self.play(Circumscribe(point[2]))
        self.play(tracker.animate.set_value(0.362))
        self.split()
        self.play(Circumscribe(point[3]))
        self.split()


class NewSim(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0, 55, 50],
            y_range=[0, 600, 100],
            axis_config={"include_numbers": True},
        ).shift(RIGHT*0.4)
        x_label = ax.get_x_axis_label(
            Tex("tosses"), edge=DOWN, direction=DOWN)
        y_label = ax.get_y_axis_label(
            Tex(r"\$"), direction=UL)
        ref = ax.get_horizontal_line(ax.c2p(50, 100), color=WHITE)

        self.play(Write(ax), Write(x_label), Write(y_label), Write(ref))

        self.split()

        coins = CoinLine(np.random.choice(['H', 'T'], 50))
        self.play(Create(coins, lag_ratio=0.1), run_time=5)
        self.play(FadeOut(coins))

        print("SIMULATING TOSSES")
        players = [100] * 500001
        avg = [0] * 51
        median = [0] * 51
        for i in range(len(avg)):
            avg[i] = np.mean(players)
            median[i] = np.median(players)
            for j in range(len(players)):
                if bool(random.getrandbits(1)):
                    players[j] *= 1 + 0.375 * 0.8
                else:
                    players[j] *= 1 - 0.375 * 0.5

        avg_graph = ax.plot_line_graph(
            x_values=range(51),
            y_values=avg,
            line_color=PURPLE_A,
            vertex_dot_radius=0.06,
            stroke_width=8
        )
        median_graph = ax.plot_line_graph(
            x_values=range(51),
            y_values=median,
            line_color=ORANGE,
            vertex_dot_radius=0.06,
            stroke_width=8
        )

        self.play(FadeIn(avg_graph))
        self.split()
        self.play(FadeIn(median_graph))
        self.split()
        value = ax.get_horizontal_line(ax.c2p(50, median[50]), color=ORANGE)
        good = Tex(
            r"final median \$" +
            "{:.0f}".format(median[50]),
            color=ORANGE
        ).next_to(ax.c2p(50, median[50]), UP).shift(UP + LEFT*0.5)
        self.play(Write(value), FadeIn(good))
        self.split()

        kelly = MathTex(
            r"f^*", "=", r"{p", r"\over", r"a}", "-", r"{q", r"\over", r"b}", font_size=60).move_to(3*UP+3*LEFT)
        kelly.set_color_by_tex("f^*", GREEN)
        kelly.set_color_by_tex("p", YELLOW)
        kelly.set_color_by_tex("a", MY_RED)
        kelly.set_color_by_tex("q", YELLOW)
        kelly.set_color_by_tex("b", MY_BLUE)

        kelly_label = Tex("Kelly Criterion!", color=ORANGE,
                          font_size=56).next_to(kelly, DOWN)

        self.play(Write(kelly))
        self.split()
        self.play(Write(kelly_label))
        self.split()
