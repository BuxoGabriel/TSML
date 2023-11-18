import { Ilayer } from "./layer";
import { Matrix } from "./matrix";
import Tensor from "./tensor";

export default class Dense extends Ilayer {
    constructor() {
        super(1, 1)
    }

    feedforward(input: Tensor): Tensor {
        return new Tensor([0])
    }

    backpropagate(error: Tensor): Tensor {
        return new Tensor([0])
    }
}