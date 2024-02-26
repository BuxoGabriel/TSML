import Tensor from "../math/tensor"
import { ALayer } from "./layer"

// Lets other models attach to this one to train with it
//
// Expected behavior is having multiple reciever models attached
// to one EmitterLayer this should generalize your emitter model
// for the various attached models. Potentially increasing its understanding
// 
// Example
// let emitter = new EmitterLayer(model1)
// let reciever1 = new RecieverLayer(model2, emitter)
// let reciever2 = new RecieverLayer(model3, emitter)
// trainer1.train(reciever1) // all 3 models are being trained
// trainer2.train(reciever2) 
// emitter.feedforward(realData)
// let reciever1Result = reciever1.getLastResult()
// let reciever2Result = reciever2.getLastResult()
export class EmitterLayer extends ALayer {
    private wrappedModel: ALayer
    private subscribers: RecieverLayer[] = []
    constructor(wrappedModel: ALayer) {
        super(wrappedModel.inputDim, wrappedModel.outputDim) 
        this.wrappedModel = wrappedModel
    }

    feedforward(input: Tensor): Tensor {
        let result = this.wrappedModel.feedforward(input)
        for(let subscriber of this.subscribers) {
            subscriber.feedforward(result)
        }
        return result
    }

    backpropagate(error: Tensor, full: boolean): Tensor {
        return this.wrappedModel.backpropagate(error, full)
    }

    applyDeltas(): void {
        this.wrappedModel.applyDeltas()
        for(let subscriber of this.subscribers) {
            subscriber.applyDeltas()
        }
    }

    subscribe(reciever: RecieverLayer) {
        this.subscribers.push(reciever)
    }
}

// Lets you attach to another model to train with it.
//
// After feedforwarding from Emitter all recievers will store
// result for access by .getLastResult()
export class RecieverLayer extends ALayer {
    private wrappedModel: ALayer
    private emitter: EmitterLayer
    private lastResult: Tensor | undefined;
    
    constructor(wrappedModel: ALayer, emitter: EmitterLayer) {
        if(wrappedModel.inputDim !== emitter.outputDim) {
            throw Error("Emitter must output tensor with same cardinality as wrappedModel input")
        }
        super(emitter.inputDim, wrappedModel.outputDim)
        this.wrappedModel = wrappedModel
        this.emitter = emitter
        emitter.subscribe(this)
    }

    feedforward(input: Tensor): Tensor {
        let result = this.emitter.feedforward(input)
        result = this.wrappedModel.feedforward(result)
        this.lastResult = result
        return result
    }

    backpropagate(error: Tensor, full: boolean = true): Tensor {
        let result = this.wrappedModel.backpropagate(error, true)
        this.emitter.backpropagate(result, full)
        return result
    }

    applyDeltas(): void {
        this.wrappedModel.applyDeltas()
    }
    
    getLastResult(): Tensor | undefined {
        return this.lastResult
    }
}
