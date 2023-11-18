import Tensor from "./tensor"

export class Matrix extends Tensor {
    public rows: number
    public cols: number

    constructor(rows: number, cols: number) {
        super([rows, cols])
        this.rows = rows
        this.cols = cols
        this.data = []
    }
    
    getData(row: number, col: number) {
        return this.data[row * this.cols + col]
    }
    
    setData(row: number, col: number, data: number) {
        this.data[row * this.cols + col] = data
    }

    static Multiply(a: Matrix, b: Matrix): Matrix {
        if(a.cols != b.rows) throw "columns of first matrix must match rows of second matrix"

        let matrix = new Matrix(a.rows, b.cols)

        for(let i = 0; i < a.rows; i++) {
            for(let j = 0; j < b.cols; j++) {
                let value = 0
                for(let k = 0; k < a.cols; j++) {
                    value += a.getData(i, k) * b.getData(k, j)
                }
                matrix.setData(i, j, value)
            }
        }

        return matrix
    }

    static fromTensor(t: Tensor): Matrix {
        let matrix: Matrix
        if(t.cardinality == 1) {
            matrix = new Matrix(t.dimensions[0], 1)
        } else if(t.cardinality = 2) {
            matrix = new Matrix(t.dimensions[0], t.dimensions[2])
        } else throw "cardinality of tensor must be 1 or 2"

        matrix.data = t.data
        return matrix
    }
}