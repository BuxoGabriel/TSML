import Conv2d from "../src/layers/conv2d"
import { activationFunction, relu } from "../src/math/func"
import Kernal from "../src/math/kernal"
import Tensor from "../src/math/tensor"

let aFn: activationFunction = {
    fn: (x) => x,
    dfn: (x) => 1
}

let conv = new Conv2d(1, 1, 0.1, aFn)

let input: Tensor = new Tensor([1, 3, 3], 
    [0, 0.5, 1,
    1.5, 2, 2.5,
    3, 3.5, 4])

let kernal: Kernal = new Kernal(1)
kernal.data = 
    [1, 0, -1,
    0, 1, 0,
    -1, 0, 1]
kernal.bias = 0.5

conv.kernals = [kernal]

let expected: Tensor = new Tensor([1, 3, 3], 
    [1, 1, 1,
    1, 1, 1,
    1, 1, 1])

let result = conv.feedforward(input)

describe("convolutional layer accuracy", () => {
    test("feedforward accuracy", () => {
        expect(result.data).toEqual(
            [2.5, 2, -0.5,
            5,   2.5,  0,
            1.5, 3, 6.5])
    })
    test("backprop accuracy", () => {
        let error = expected.subtract(result)
        expect(error.data).toEqual(
        [-1.5, -1, 1.5,
        -4, -1.5, 1,
        -0.5, -2, -5.5])

        let inputErr = conv.backpropagate(error, true)

        let expectedDelta = (error.data[4] * input.data[0] + error.data[5] * input.data[1] + error.data[7] * input.data[3] + error.data[8] * input.data[4]) / 9 * conv.learningRate
        expect(conv.kernalDeltas[0].data[0]).toEqual(expectedDelta)
    })
})