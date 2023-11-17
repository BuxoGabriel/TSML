export class Matrix {
    public rows: number
    public cols: number
    public data: number[][]

    constructor(rows: number, cols: number) {
        this.rows = rows
        this.cols = cols
        this.data = []
        for (let i = 0; i < rows; i++) {
            this.data[i] = []
            for (let j = 0; j < cols; j++) {
                this.data[i][j] = 0
            }
        }
    }


}