import Tensor from "../math/tensor";

export abstract class ALayer {
    public lr: number = 0.01
    public inputDim: number
    public outputDim: number
    public cost: number = 0

    constructor(inputDim: number, outputDim: number) {
        this.inputDim = inputDim
        this.outputDim = outputDim
    }

    abstract feedforward(input: Tensor): Tensor
    abstract backpropagate(error: Tensor, full: boolean): Tensor
    abstract applyDeltas(): void

    train(input: Tensor, expected: Tensor, applyDeltas: boolean = true): number {
        let result: Tensor = this.feedforward(input)
        let error = expected.subtract(result)
        this.cost = error.map(x => x * x / 2 / error.size * 100, false).sum()

        this.backpropagate(error, false)
        if(applyDeltas) this.applyDeltas()
        return this.cost
    }

    batchTrain(input: Tensor[], expected: Tensor[]): number {
        let len = input.length
        let cost = 0
        if(expected.length !== len) throw Error("Input and expected arrays must be same sized for batch training")
        for(let i = 0; i < len; i++) {
            cost += this.train(input[i], expected[i], false) / len
        }
        this.applyDeltas()
        this.cost = cost
        return cost
    }

    accepts(input: Tensor): boolean {
        if (input.cardinality == this.inputDim) return true
        else return false
    }

    acceptsError(error: Tensor): boolean {
        if (error.cardinality == this.outputDim) return true
        else return false
    }
}

export class CompositeLayer extends Alayer {
    private layers: Alayer[]

    constructor() {
        super(0, 0)
        this.layers = []
    }

    addLayer(layer: Alayer): CompositeLayer {
        if (!this.layers.length) {
            this.inputDim = layer.inputDim
        } else if (this.outputDim != layer.inputDim) {
            throw new Error("Input dimensions of added layer must match output dimensions of existing composite layer")
        }

        this.outputDim = layer.outputDim
        this.layers.push(layer)
        return this
    }

    feedforward(input: Tensor): Tensor {
        let output: Tensor = input
        for (let layer of this.layers) {
            output = layer.feedforward(output)
        }
        return output
    }

    backpropagate(error: Tensor, full: boolean = true): Tensor {
        for (let i = this.layers.length - 1; i >= 0; i--) {
            if (error.cardinality != this.layers[i].outputDim) throw new Error(`Cardinality of error does not match outputDim for layer ${i}`)
            error = (full || i != 0) ? this.layers[i].backpropagate(error, true) : this.layers[i].backpropagate(error, false)
        }
        return error
    }

    applyDeltas(): void {
        for (let layer of this.layers) {
            layer.applyDeltas()
        }
    }
}
