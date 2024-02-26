import { ALayer } from "./layer";
import Matrix from "../math/matrix";
import Tensor from "../math/tensor";

export default class Flatten extends ALayer {
    private inputSignature: number[]
    constructor(inputSignature: number[]) {
        super(3, 2)
        this.inputSignature = inputSignature
    }

    feedforward(input: Tensor): Matrix {
        if (!this.accepts(input)) throw Error("Flatten layer only supports 3 dimensional tensors as inputs")
        return new Matrix(input.size, 1, input.data)
    }

    backpropagate(error: Tensor, full: boolean): Tensor {
        if (!this.acceptsError(error)) throw Error("Flatten layer only accepts 2d tensors or matrixes as error")
        return new Tensor(this.inputSignature, error.data)
    }

    applyDeltas(): void {
        return
    }
}
