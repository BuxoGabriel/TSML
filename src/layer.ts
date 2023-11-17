import Tensor from "./tensor";

export abstract class Ilayer {
    public inputDim: number
    public outputDim: number

    constructor(inputDim: number, outputDim: number) {
        this.inputDim = inputDim
        this.outputDim = outputDim
    }

    abstract feedforward(input: Tensor): Tensor
    abstract backpropagate(error: Tensor): Tensor

    train(input: Tensor, expected: Tensor): void {
        let result: Tensor = this.feedforward(input)
        let error = Tensor.subtract(result, expected)
        this.backpropagate(error)
    }
}

export class CompositeLayer extends Ilayer {
    private lr: number
    private layers: Ilayer[]

    constructor() {
        super(0, 0)
        this.lr = 1
        this.layers = []
    }

    addLayer(layer: Ilayer): CompositeLayer {
        if(!this.layers.length) {
            this.inputDim = layer.inputDim
        } else if(this.outputDim != layer.inputDim) {
            throw "Input dimensions of added layer must match output dimensions of existing composite layer"
        }

        this.outputDim = layer.outputDim
        this.layers.push(layer)
        return this
    }

    feedforward(input: Tensor): Tensor {
        let output: Tensor = input
        for(let layer of this.layers) {
            output = layer.feedforward(output)
        }
        return output
    }

    backpropagate(error: Tensor): Tensor {
        for(let i = this.layers.length - 1; i >= 0; i--) {
            if(error.cardinality != this.layers[i].outputDim) throw `Cardinality of error does not match outputDim for layer ${i}`
            error = this.layers[i].backpropagate(error)
        }
        return error
    }
}