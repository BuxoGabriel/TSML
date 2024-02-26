import { ALayer } from "./layer";
import Kernal from "../math/kernal";
import Tensor from "../math/tensor";
import { activationFunction, sigmoid } from "../math/func";

export default class Conv2d extends ALayer {
    input?: Tensor
    kernals: Kernal[] = []
    learningRate: number
    numKernals: number
    kernalRadius: number
    kernalDeltas: Kernal[] = []
    aFn: activationFunction

    constructor(numKernals: number, kernalRadius: number, learningRate: number = 0.1, activationFunction: activationFunction = sigmoid) {
        super(3, 3)
        this.numKernals = numKernals
        this.kernalRadius = kernalRadius
        this.learningRate = learningRate
        this.aFn = activationFunction
        for(let i = 0; i < numKernals; i++) {
            this.kernals.push(new Kernal(kernalRadius))
            this.kernalDeltas.push(new Kernal(kernalRadius).zero())
        }
    }

    // each kernal passes on each image sequentially so kernal 1 image 1, kernal 1 image 2, kernal 2 image 1, kernal 2 image 2
    feedforward(input: Tensor): Tensor {
        if(!this.accepts(input)) throw Error("Input tensor for convolutional layer must be 3 dimensional")
        this.input = input
        const [depth, rows, cols] = input.dimensions
        let output = new Tensor([depth * this.numKernals, rows, cols])
        for(let i = 0; i < this.numKernals; i++) {
            output = this.kernals[i].pass(input, output, i)
        }
        output = output.map(this.aFn.fn, true)
        return output
    }

    backpropagate(error: Tensor, full: boolean): Tensor {
        if(!this.input) throw new Error("Can only backprop after feedforward, input not detected")
        let [depth, rows, cols] = this.input.dimensions
        let inputErr = new Tensor(this.input.dimensions)
        let imageSize = rows * cols

        for(let image = 0; image < depth; image++) {
            for(let row = 0; row < rows; row++) {
                for(let col = 0; col < cols; col++) {
                    // for each pixel of every input image
                    let index = (image * rows * cols) + (row * cols) + col

                    for(let kernal = 0; kernal < this.numKernals; kernal++) {
                        let k = this.kernals[kernal]
                        let kd = this.kernalDeltas[kernal]
                        let pixelErr = this.aFn.dfn(error.data[kernal * this.input.size + index]) * error.data[kernal * this.input.size + index]
                        for(let yOff = -k.radius; yOff <= k.radius; yOff++) {
                            for(let xOff = -k.radius; xOff <= k.radius; xOff++) {
                                // for each pixel of input that contributed to result
                                let y = row + yOff
                                let x = col + xOff
                                let ky = yOff + k.radius
                                let kx = xOff + k.radius
                                if(y < 0 || y >= rows) continue
                                if(x < 0 || x >= cols) continue
                                let inputIndex = image * rows * cols + y * cols + x
                                let addedDelta =  pixelErr * this.input.data[inputIndex] * this.learningRate / imageSize
                                kd.setData(ky, kx, kd.getData(ky, kx) + addedDelta)
                                inputErr.data[inputIndex] += k.getData(ky, kx) * pixelErr * this.learningRate / imageSize
                            }
                        }
                    }
                }
            }
        }
        return inputErr
    }

    applyDeltas(): void {
        for(let i = 0; i < this.numKernals; i++) {
            this.kernals[i].add(this.kernalDeltas[i], true)
            this.kernalDeltas[i].zero()
            this.kernalDeltas[i].bias = 0
        }
    }
}
