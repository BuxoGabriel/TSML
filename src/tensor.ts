export default class Tensor {
    public cardinality: number
    constructor() {
        this.cardinality = 0
    }

    static subtract(a: Tensor, b: Tensor): Tensor {
        return new Tensor()
    }
}