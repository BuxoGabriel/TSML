import Tensor from "./tensor"

export default class Matrix extends Tensor {
    public rows: number
    public cols: number

    constructor(rows: number, cols: number, data?: number[]) {
        if(data && data.length != rows * cols) throw new Error("length of data must exactly match size of matrix")
        else super([rows, cols], data)
        this.rows = rows
        this.cols = cols
    }
    
    getData(row: number, col: number): number {
        if(row < 0 || row >= this.rows) throw new Error("row out of bounds")
        if(col < 0 || col >= this.cols) throw new Error("col out of bounds")
        return this.data[row * this.cols + col]
    }
    
    setData(row: number, col: number, data: number): void {
        if(row < 0 || row >= this.rows) throw new Error("row out of bounds")
        if(col < 0 || col >= this.cols) throw new Error("col out of bounds")
        this.data[row * this.cols + col] = data
    }

    // transposition function
    transpose(): Matrix {
        if(this.rows == this.cols) {
            for(let row = 0; row < this.rows; row++) {
                for(let col = row + 1; col < this.cols; col++) {
                    [this.data[row * this.cols + col], this.data[col * this.rows + row]] = [this.getData(col, row), this.getData(row, col)]
                }
            }
        } else {
            let transposeData = []
            for(let col = 0; col < this.cols; col++) {
                for(let row = 0; row < this.rows; row++) {
                    transposeData.push(this.getData(row, col))
                }
            }
            this.data = transposeData
        }
        [this.rows, this.cols] = [this.cols, this.rows]
        return this
    }

    randomize(low: number = 0, high: number = 1, inPlace: boolean = false, floor: boolean = false): Matrix {
        let tensor: Tensor = super.randomize(low, high, inPlace, floor)
        let matrix: Matrix
        if(!inPlace) matrix = Matrix.fromTensor(tensor)
        else matrix = tensor as Matrix
        return matrix
    }

    static createTranspose(a: Matrix): Matrix {
        let matrix = new Matrix(a.cols, a.rows)
        for(let i = 0; i < a.rows; i++) {
            for(let j = 0; j < a.cols; j++) {
                matrix.setData(j, i, a.getData(i, j))
            }
        }
        return matrix
    }

    static Multiply(a: Matrix, b: Matrix): Matrix {
        if(a.cols != b.rows) throw new Error("columns of first matrix must match rows of second matrix")

        let matrix = new Matrix(a.rows, b.cols)

        for(let i = 0; i < a.rows; i++) {
            for(let j = 0; j < b.cols; j++) {
                let value = 0
                for(let k = 0; k < a.cols; k++) {
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
        } else if(t.cardinality == 2) {
            matrix = new Matrix(t.dimensions[0], t.dimensions[1])
        } else throw new Error("cardinality of tensor must be 1 or 2")

        matrix.data = t.data
        return matrix
    }
}
