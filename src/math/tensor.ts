export default class Tensor {
    public dimensions: number[]
    public cardinality: number
    public size: number
    public data: number[]
    constructor(dimensions: number[], data?: number[]) {
        if(dimensions.length == 0) throw new Error("tensor can not be 0 dimensional")
        this.dimensions = dimensions
        this.cardinality = dimensions.length
        this.size = 1

        for(let dim of dimensions) {
            if(dim < 1) throw new Error("tensor can not have dimension with size less than 1")
            this.size *= dim
        }
        if(data){
            if(data.length != this.size) throw new Error("length of data must exactly match size of tensor")
            this.data = data
        }
        else this.data = new Array(this.size).fill(0)
    }

    matchesSignature(other: Tensor): boolean {
        if(this.cardinality != other.cardinality) return false
        for(let i = 0; i < this.cardinality; i++) {
            if(this.dimensions[i] != other.dimensions[i]) return false
        }
        return true
    }

    operation(other: Tensor, fn: (a: number, b: number) => number, inPlace = false) {
        if(!this.matchesSignature(other)) throw new Error("dimensions of tensors must match for subtraction")

        let tensor: Tensor
        if(inPlace) tensor = this
        else tensor = new Tensor(this.dimensions)

        for(let i = 0; i < this.size; i++) {
            tensor.data[i] = fn(this.data[i], other.data[i])
        }

        return tensor
    }

    add(other: Tensor, inPlace: boolean = false): Tensor {
        return this.operation(other, (a, b) => a + b, inPlace)
    }

    subtract(other: Tensor, inPlace: boolean = false): Tensor {
        return this.operation(other, (a, b) => a - b, inPlace)
    }
    
    piecewiseMultiply(other: Tensor, inPlace: boolean = false): Tensor {
        return this.operation(other, (a, b) => a * b, inPlace)
    }

    map(fn: (x: number) => number, inPlace: boolean = false): Tensor {
        let tensor: Tensor
        if(inPlace) tensor = this
        else tensor = new Tensor(this.dimensions)
        for(let i = 0; i < this.size; i++) {
            tensor.data[i] = fn(this.data[i])
        }
        return tensor
    }

    randomize(low: number = 0, high: number = 1, inPlace: boolean = false, floor: boolean = false): Tensor {
        return this.map(_ => {
            let x = Math.random() * (high - low) + low;
            return floor? Math.floor(x): x
        }, inPlace)
    }

    zero(): Tensor {
        return this.map(_ => 0, true)
    }

    sum(): number {
        let total = 0
        for(let i = 0; i < this.size; i++) {
            total += this.data[i]
        }
        return total
    }
 
    static clone(t: Tensor) {
        let tensor = new Tensor(t.dimensions)
        tensor.operation(t, (a, b) => b, true)
        return tensor
    }
}