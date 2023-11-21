import Dense from "../src/dense"
import { activationFunction, relu } from "../src/func"
import Matrix from "../src/matrix"

describe("dense layer math tests", () => {
    test("feedforward", () => {
        const learningRate = 0.1;
        const activationFunction: activationFunction = {
            fn: x => x,
            dfn: x => 1
        }

        let NN = new Dense([2, 1], learningRate, activationFunction)
        NN.setParameters({weights: [new Matrix(1, 2, [3, 1])], biases: [new Matrix(1, 1, [0.5])]})
        const input = new Matrix(2, 1, [0.5, 1.5])
        const output = NN.feedforward(input)
        const layers = NN.getLayers()
        const answer = activationFunction.fn(0.5 * 3 + 1.5 * 1 + 0.5)
        expect(layers[0].data).toEqual([0.5, 1.5])
        expect(layers[1].data).toEqual([answer])

        const expectedValue = new Matrix(1, 1, [1])
        const error = expectedValue.subtract(output)
        expect(error.data).toEqual([1 - answer])
        const inputErr = Matrix.fromTensor(NN.backpropagate(error))
        const { weightDeltas, biasDeltas } = NN.getDeltas()
        expect(biasDeltas[0].data).toEqual([activationFunction.dfn(answer) * (1 - answer) * learningRate])
        expect(weightDeltas[0].data).toEqual([activationFunction.dfn(answer) * (1 - answer) * 0.5 * learningRate, activationFunction.dfn(answer) * (1 - answer) * 1.5 * learningRate])
        expect(inputErr.data).toEqual([activationFunction.dfn(answer) * (1 - answer) * 3, activationFunction.dfn(answer) * (1 - answer) * 1])
    })
})