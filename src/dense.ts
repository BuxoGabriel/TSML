import { activationFunction, lossFunction, sigmoid, squareErr } from "./func";
import { Alayer } from "./layer";
import Matrix from "./matrix";
import Tensor from "./tensor";

export default class Dense extends Alayer {
    private aFn: activationFunction
    private learningRate: number
    private structure: number[]

    private layers: Matrix[]
    private weights: Matrix[]
    private biases: Matrix[]
    private weightDeltas: Matrix[]
    private biasDeltas: Matrix[]
    constructor(structure: number[], learningRate: number = 0.1, aFn: activationFunction = sigmoid) {
        super(2, 2)
        this.structure = structure
        this.aFn = aFn
        this.learningRate = learningRate

        this.layers = [new Matrix(structure[0], 1)]
        this.weights = []
        this.biases = []
        this.weightDeltas = []
        this.biasDeltas = []

        for (let i = 1; i < structure.length; i++) {
            this.weights.push(new Matrix(structure[i], structure[i - 1]).randomize(0, 1, true))
            this.biases.push(new Matrix(structure[i], 1).randomize(0, 1, true))
            this.weightDeltas.push(new Matrix(structure[i], structure[i - 1]))
            this.biasDeltas.push(new Matrix(structure[i], 1))
        }
    }

    feedforward(input: Matrix | Tensor): Matrix {
        if (!this.accepts(input)) throw Error("Dense layer only supports 2 dimensional tensors as inputs")
        // set input layer
        this.layers[0].data = input.data
        if(!(this.layers[0].cols == 1)) throw Error("input must be 1d/column Matrix")

        for (let i = 0; i < this.weights.length; i++) {
            // H(n + 1) = Afn(W(n)H(n) + B(n))
            this.layers[i + 1] = Matrix.Multiply(this.weights[i], this.layers[i])
            this.layers[i + 1].add(this.biases[i], true)
            this.layers[i + 1].map(this.aFn.fn, true)
        }
        return this.layers[this.layers.length - 1]
    }

    backpropagate(error: Tensor, full: boolean = true): Tensor {
        // accepts 2d tensor or Matrix, must be column
        if (!this.acceptsError(error)) throw Error("Dense layer only supports 1 dimensional tensors as error")
        if (!(error instanceof Matrix)) error = Matrix.fromTensor(error)
        let mError = error as Matrix
        if(!(mError.cols == 1)) throw Error("error must be 1d/column Matrix")

        for (let i = this.weights.length - 1; i >= 0; i--) {
            mError.piecewiseMultiply(this.layers[i + 1].map(this.aFn.dfn, false), true)
            let dW = Matrix.Multiply(mError, Matrix.createTranspose(this.layers[i]))
            this.weightDeltas[i].add(dW.map(x => x * this.learningRate), true)
            this.biasDeltas[i].add(mError.map(x => x * this.learningRate), true)
            if(i == 0 && !full) break
            mError = Matrix.Multiply(Matrix.createTranspose(this.weights[i]), mError) as Matrix
            mError.map(x =>  x / this.layers[i + 1].rows)
        }
        return mError
    }

    applyDeltas(): void {
        this.weights.forEach((w, index) => {
            w.add(this.weightDeltas[index], true)
            this.weightDeltas[index].zero()
        })
        this.biases.forEach((b, index) => {
            b.add(this.biasDeltas[index], true)
            this.biasDeltas[index].zero()
        })

    }

    setParameters({ weights, biases }: { weights?: Matrix[], biases: Matrix[] }) {
        if (weights) this.weights = weights
        if (biases) this.biases = biases
    }

    getParameters(): { weights: Matrix[], biases: Matrix[] } {
        return {
            weights: this.weights,
            biases: this.biases
        }
    }

    getLayers(): Matrix[] {
        return this.layers
    }

    getDeltas(): { weightDeltas: Matrix[], biasDeltas: Matrix[] } {
        return {
            weightDeltas: this.weightDeltas,
            biasDeltas: this.biasDeltas
        }
    }
}
