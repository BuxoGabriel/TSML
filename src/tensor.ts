export default class Tensor {
    public dimensions: number[]
    public cardinality: number
    public size: number
    public data: number[]
    constructor(dimensions: number[]) {
        this.dimensions = dimensions
        this.cardinality = dimensions.length
        this.size = 1

        for(let dim of dimensions) {
            this.size *= dim
        }

        this.data = new Array(this.size).fill(0)
    }

    matchesSignature(other: Tensor): boolean {
        if(this.cardinality != other.cardinality) return false
        for(let i = 0; i < this.cardinality; i++) {
            if(this.dimensions[i] != other.dimensions[i]) return false
        }
        return true
    }

    operation(other: Tensor, fn: (a: number, b: number) => number, inPlace = false) {
        if(!this.matchesSignature(other)) throw "dimensions of tensors must match for subtraction"

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
    
    piecewiseWultiply(other: Tensor, inPlace: boolean = false): Tensor {
        return this.operation(other, (a, b) => a * b, inPlace)
    }
}