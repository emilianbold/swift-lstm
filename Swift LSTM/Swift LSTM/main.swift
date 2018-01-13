// g(t) = ϕ(Wgx x(t) +Wgh h(t−1) + bg)
// i(t) = σ(Wix x(t) + Wih h(t−1) + bi)
// f(t) = σ(Wfx x(t) + Wfh h(t−1) + bf)
// o(t) = σ(Wox x(t) + Woh h(t−1) + bo)
// s(t) = g(t) ∗ i(t) + s(t−1) ∗ f(t)
import Foundation

class Math {
    static func sigmoid(_ x : Array<Double>) -> Array<Double> {
        return x.map { v in sigmoid(v) }
    }

    static func sigmoid(_ x : Double) -> Double {
        return 1.0 / (1.0 + exp(-x))
    }

    static func sigmoid_derivative(_ x : Array<Double>) -> Array<Double> {
        return x.map { v in sigmoid_derivative(v) }
    }

    static func sigmoid_derivative(_ x : Double) -> Double {
        return x * (1 - x)
    }

    static func _tanh(_ x : Array<Double>) -> Array<Double> {
        return x.map { v in tanh(v) }
    }

    static func tanh_derivative(_ x : Array<Double>) -> Array<Double> {
        return x.map { v in 1 - v * v}
    }

    static func _tanh(x:Double) -> Double {
        return tanh(x)
    }

}

infix operator  .+
infix operator  .-
infix operator  .*

extension Array where Element == Double {

    init(random min: Double, max: Double, count: Int) {
        self.init(repeating: 0.0, count: count)
        for i in 0...count-1 {
            self[i] = min + (max - min) * (Double(arc4random_uniform(1000)) / 1000.0)
        }
    }

    static func .+ (_ a: Array<Double>, _ b : Array<Double>) -> Array<Double> {
        var c = Array<Double>(a)
        for i in 0...c.count-1 {
            c[i] = c[i] + b[i]
        }

        return c
    }

    static func .- (_ a: Array<Double>, _ b : Array<Double>) -> Array<Double> {
        var c = Array<Double>(a)
        for i in 0...c.count-1 {
            c[i] = c[i] - b[i]
        }

        return c
    }

    static func .* (_ a: Array<Double>, _ b : Array<Double>) -> Array<Double> {
        var c = Array<Double>(a)
        for i in 0...c.count-1 {
            c[i] = c[i] * b[i]
        }

        return c
    }

    static func .* (a: Array<Double>, b: Double) -> Array<Double> {
        var c = Array<Double>(a)
        for i in 0...c.count-1 {
            c[i] = c[i] * b
        }

        return c
    }

    static func += (a: inout Array<Double>, b: Array<Double>) {
        for i in 0...a.count-1 {
            a[i] = a[i] + b[i]
        }
    }

    static func -= (a: inout Array<Double>, b: Array<Double>) {
        for i in 0...a.count-1 {
            a[i] = a[i] - b[i]
        }
    }
}

func directAccessor(rows : Int) -> ((Int, Int) -> Int) {
    //by row, 0-indexed
    return { (r, c) in (c * rows + r) }
}

class Matrix<Element> {
    var data : Array<Element>
    let rows, cols : Int
    let accessor : (Int, Int) -> Int
    let streamOK : Bool

    convenience init(rows : Int, cols: Int, data : Array<Element>) {
        self.init(rows: rows, cols: cols, data: data, accessor: directAccessor(rows: rows), streamOK: true)
    }

    init(rows : Int, cols: Int, data : Array<Element>, accessor: @escaping (Int, Int) -> Int, streamOK: Bool) {
        self.data = data
        self.rows = rows
        self.cols = cols
        self.accessor = accessor
        self.streamOK = streamOK
    }

    init(repeating v: Element, rows : Int, cols: Int) {
        self.data = Array(repeating: v, count: rows * cols)
        self.rows = rows
        self.cols = cols
        self.accessor = directAccessor(rows: rows)
        self.streamOK = true
    }

    func transpose() -> Matrix<Element> {
        return Matrix(rows: cols, cols: rows, data: data, accessor: { (i,j) in self.accessor(j,i) }, streamOK: !self.streamOK)
    }

    func stream() -> Array<Element> {
        assert(streamOK, "Transposed matrix should have the stream in reverse, which is not implemented yet")
        return data
    }

    subscript(i : Int, j : Int) -> Element {
        assert(!(i<0 || i >= rows || j < 0 || j >= cols))
        return data[accessor(i, j)]
    }

}

infix operator §

extension Matrix where Element == Double {
    convenience init(random min: Double, max: Double, rows: Int, cols: Int) {
        self.init(rows: rows, cols: cols, data: Array(random: min, max: max, count: rows * cols))
    }

    static func § (_ a: Matrix<Double>, _ b : Array<Double>) -> Array <Double> {
        assert(a.cols == b.count)
        var r = [Double](repeating: 0.0, count: a.rows)
        for row in 0...a.rows-1 {
            var sum = 0.0
            for col in 0...a.cols-1 {
                sum += b[col] * a[row, col]
            }
            r[row] = sum
        }

        return r
    }

    static func += (left: inout Matrix<Double>, right: Matrix<Double>) {
        assert(left.rows == right.rows && left.cols == right.cols)
        assert(left.streamOK && right.streamOK)
        left.data += right.data
//        left = Matrix(rows: left.rows, cols: left.cols, data: (left.stream() .+ right.stream()), accessor: directAccessor(rows: left.rows), streamOK: true)
    }

    static func -= (left: inout Matrix<Double>, right: Matrix<Double>) {
        assert(left.rows == right.rows && left.cols == right.cols)
        assert(left.streamOK && right.streamOK)
        left.data -= right.data
//        left = Matrix(rows: left.rows, cols: left.cols, data: (left.stream() .- right.stream()), accessor: directAccessor(rows: left.rows), streamOK: true)
    }

    static func .* (left: Matrix<Double>, right: Double) -> Matrix<Double> {
        return Matrix(rows: left.rows, cols: left.cols, data: (left.stream() .* right))
    }

    static func outer(_ a: [Double], _ b: [Double]) -> Matrix<Double> {
        var data = [Double](repeating: 0, count : a.count * b.count)
        let accessor = directAccessor(rows: a.count)
        for row in 0...a.count-1 {
            for col in 0...b.count-1 {
                data[accessor(row, col)] = a[row] * b[col]
            }
        }
        return Matrix(rows: a.count, cols: b.count, data: data, accessor: accessor, streamOK: true)
    }

    static func zero_like(_ a : Matrix<Double>) -> Matrix<Double> {
        return Matrix(repeating: 0.0, rows: a.rows, cols: a.cols)
    }

}

class WeightsAndBiases {
    var wg, wi, wf, wo : Matrix<Double>
    var bg, bi, bf, bo : Array<Double>
    let inputSize, cellsSize, hiddenEdgesSize : Int
    var wg_diff, wi_diff, wf_diff, wo_diff : Matrix<Double>
    var bg_diff, bi_diff, bf_diff, bo_diff : Array<Double>

    init(cellsSize : Int, inputSize : Int, randomMin: Double, randomMax: Double) {
        self.inputSize = inputSize
        self.cellsSize = cellsSize
        self.hiddenEdgesSize = cellsSize + inputSize

        self.wg = Matrix(random: randomMin, max: randomMax, rows: cellsSize, cols: hiddenEdgesSize)
        self.wi = Matrix(random: randomMin, max: randomMax, rows: cellsSize, cols: hiddenEdgesSize)
        self.wf = Matrix(random: randomMin, max: randomMax, rows: cellsSize, cols: hiddenEdgesSize)
        self.wo = Matrix(random: randomMin, max: randomMax, rows: cellsSize, cols: hiddenEdgesSize)

        self.bg = Array(random: randomMin, max: randomMax, count: cellsSize)
        self.bi = Array(random: randomMin, max: randomMax, count: cellsSize)
        self.bf = Array(random: randomMin, max: randomMax, count: cellsSize)
        self.bo = Array(random: randomMin, max: randomMax, count: cellsSize)

        // diffs (derivative of loss function w.r.t. all parameters)
        self.wg_diff = Matrix(repeating: 0.0, rows: cellsSize, cols: hiddenEdgesSize)
        self.wi_diff = Matrix(repeating: 0.0, rows: cellsSize, cols: hiddenEdgesSize)
        self.wf_diff = Matrix(repeating: 0.0, rows: cellsSize, cols: hiddenEdgesSize)
        self.wo_diff = Matrix(repeating: 0.0, rows: cellsSize, cols: hiddenEdgesSize)
        self.bg_diff = Array(repeating: 0, count: cellsSize)
        self.bi_diff = Array(repeating: 0, count: cellsSize)
        self.bf_diff = Array(repeating: 0, count: cellsSize)
        self.bo_diff = Array(repeating: 0, count: cellsSize)
    }

    func apply_diff(_ lr : Double) {
        self.wg -= self.wg_diff .* lr
        self.wi -= self.wi_diff .* lr
        self.wf -= self.wf_diff .* lr
        self.wo -= self.wo_diff .* lr
        self.bg -= self.bg_diff .* lr
        self.bi -= self.bi_diff .* lr
        self.bf -= self.bf_diff .* lr
        self.bo -= self.bo_diff .* lr

        self.wg_diff = Matrix(repeating: 0.0, rows: cellsSize, cols: hiddenEdgesSize)
        self.wi_diff = Matrix(repeating: 0.0, rows: cellsSize, cols: hiddenEdgesSize)
        self.wf_diff = Matrix(repeating: 0.0, rows: cellsSize, cols: hiddenEdgesSize)
        self.wo_diff = Matrix(repeating: 0.0, rows: cellsSize, cols: hiddenEdgesSize)
        self.bg_diff = Array(repeating: 0, count: cellsSize)
        self.bi_diff = Array(repeating: 0, count: cellsSize)
        self.bf_diff = Array(repeating: 0, count: cellsSize)
        self.bo_diff = Array(repeating: 0, count: cellsSize)
    }
}

class State {
    var s, h : Array<Double>
    var g, i, f, o : Array<Double>
    var bottom_diff_h, bottom_diff_s : Array<Double>

    init(cellsSize : Int) {
        self.g = Array(repeating: 0, count: cellsSize)
        self.i = Array(repeating: 0, count: cellsSize)
        self.f = Array(repeating: 0, count: cellsSize)
        self.o = Array(repeating: 0, count: cellsSize)
        self.s = Array(repeating: 0, count: cellsSize)
        self.h = Array(repeating: 0, count: cellsSize)
        self.bottom_diff_h = Array(repeating: 0, count: cellsSize)
        self.bottom_diff_s = Array(repeating: 0, count: cellsSize)
    }

}

class Node {
    let state : State
    let param : WeightsAndBiases
    var s_prev, h_prev : Array<Double>
    var xc : Array<Double>

    init(_ param: WeightsAndBiases, _ state: State) {
        self.state = state
        self.param = param
        //TODO: proper dimensions?
        self.s_prev = []
        self.h_prev = []
        self.xc = []
    }

    func bottom_data_is(_ x : Array<Double>, _ s_prev : Array<Double>?, _ h_prev : Array<Double>?) {
        self.s_prev = s_prev ?? Array(repeating: 0, count: self.state.s.count)
        self.h_prev = h_prev ?? Array(repeating: 0, count: self.state.h.count)

        // concatenate x(t) and h(t-1)
        let xc = x + self.h_prev

        self.state.g = Math._tanh((self.param.wg § xc) .+ self.param.bg)
        self.state.i = Math.sigmoid((self.param.wi § xc) .+ self.param.bi)
        self.state.f = Math.sigmoid((self.param.wf § xc) .+ self.param.bf)
        self.state.o = Math.sigmoid((self.param.wo § xc) .+ self.param.bo)
        self.state.s = (self.state.g .* self.state.i) .+ (self.s_prev .* self.state.f)
        self.state.h = self.state.s .* self.state.o

        self.xc = xc
    }

    func top_diff_is(_ top_diff_h : Array<Double>, _ top_diff_s : Array<Double>) {
        // notice that top_diff_s is carried along the constant error carousel
        let ds = (self.state.o .* top_diff_h) .+ top_diff_s
        let _do = self.state.s .* top_diff_h
        let di = self.state.g .* ds
        let dg = self.state.i .* ds
        let df = self.s_prev .* ds

        // diffs w.r.t. vector inside sigma / tanh function
        let di_input = Math.sigmoid_derivative(self.state.i) .* di
        let df_input = Math.sigmoid_derivative(self.state.f) .* df
        let do_input = Math.sigmoid_derivative(self.state.o) .* _do
        let dg_input = Math.tanh_derivative(self.state.g) .* dg

        // diffs w.r.t. inputs
        self.param.wi_diff += Matrix.outer(di_input, self.xc)
        self.param.wf_diff += Matrix.outer(df_input, self.xc)
        self.param.wo_diff += Matrix.outer(do_input, self.xc)
        self.param.wg_diff += Matrix.outer(dg_input, self.xc)
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input

        // compute bottom diff
        var dxc = [Double](repeating: 0.0, count: self.xc.count)
        dxc += self.param.wi.transpose() § di_input
        dxc += self.param.wf.transpose() § df_input
        dxc += self.param.wo.transpose() § do_input
        dxc += self.param.wg.transpose() § dg_input

        // save bottom diffs
        self.state.bottom_diff_s = ds .* self.state.f
        self.state.bottom_diff_h = Array.init(dxc.suffix(self.param.cellsSize))
    }
}

class Network {
    var inputs : Array<Array<Double>>
    var nodes : Array<Node>
    var param: WeightsAndBiases

    init(param: WeightsAndBiases) {
        self.param = param
        self.inputs = []
        self.nodes = []
    }

    func x_list_clear() {
        inputs.removeAll()
    }

    func x_list_add(_ x : Array<Double>) {
        let prevNode : Node?
        if inputs.isEmpty {
            prevNode = Optional.none
        } else {
            prevNode = nodes[inputs.count - 1]
        }

        inputs.append(x)

        let node : Node
        if(inputs.count > nodes.count) {
            let state = State(cellsSize: param.cellsSize)
            node = Node(param, state)
            nodes.append(node)
        } else {
            node = nodes[inputs.count - 1]
        }

        node.bottom_data_is(x, prevNode?.state.s, prevNode?.state.h)
    }

    func y_list_is(_ y_list: Array<Double>, lossF: (Array<Double>, Double) -> Double, bottom_diff: (Array<Double>, Double) -> Array<Double>) -> Double {
        var idx = self.inputs.count - 1

        var loss = lossF(self.nodes[idx].state.h, y_list[idx])
        let diff_h = bottom_diff(self.nodes[idx].state.h, y_list[idx])
        // here s is not affecting loss due to h(t+1), hence we set equal to zero
        self.nodes[idx].top_diff_is(diff_h, [Double](repeating: 0.0, count: self.param.cellsSize))
        idx = idx - 1

        // ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
        // we also propagate error along constant error carousel using diff_s
        while idx >= 0 {
            loss += lossF(self.nodes[idx].state.h, y_list[idx])
            let diff_h = bottom_diff(self.nodes[idx].state.h, y_list[idx])
                            .+ self.nodes[idx + 1].state.bottom_diff_h
            let diff_s = self.nodes[idx + 1].state.bottom_diff_s
            self.nodes[idx].top_diff_is(diff_h, diff_s)

            idx = idx - 1
        }

        return loss
    }
}

// Config
// Note: full back propagation through time!
let maxIteration    = 100

//rnn.convergenceError = config.convergenceError.doubleValue;
let learningRate = 0.1
let randomMax        = 0.25
let randomMin        = randomMax * -1

let hiddenLayerSize = 100
// ... done config.

let x_dim = 50
let lstm_param = WeightsAndBiases(cellsSize: hiddenLayerSize, inputSize: x_dim, randomMin: randomMin, randomMax: randomMax)
let lstm_net = Network(param: lstm_param)
let y_list = [-0.5, 0.2, 0.1, -0.5]
let input_val_arr = y_list.map { _ in Array(random: 0, max:1, count: x_dim) }

for cur_iter in 0...maxIteration-1 {
    print("iter", cur_iter)

    for ind in 0...y_list.count-1 {
        lstm_net.x_list_add(input_val_arr[ind])
    }


    for ind in 0...y_list.count-1 {
        print("y_pred = [",  lstm_net.nodes[ind].state.h[0])
    }

    let loss = lstm_net.y_list_is(y_list,
                                  lossF: { (hh, y) in  (hh[0] - y) * (hh[0] - y)},
                                  bottom_diff: { (p, l) in
                                    var diff = [Double](repeating: 0, count: p.count)

                                    diff[0] = 2 * (p[0] - l)
                                    return diff
                                    })
    print("loss:", loss)
    lstm_param.apply_diff(learningRate)
    lstm_net.x_list_clear()
}
