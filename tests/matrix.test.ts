import { describe, expect, test } from "@jest/globals"
import Matrix from "../src/math/matrix"
import Tensor from "../src/math/tensor"

describe("matrix module", () => {
    test("Matrix initialization", () => {
        let a = new Matrix(3, 2)
        expect(a.data).toEqual([0, 0, 0, 0, 0, 0])

        const data = [1, 2, 3, 4, 5, 6]
        a = new Matrix(3, 2, data)
        expect(a.data).toBe(data)

        expect(() => new Matrix(2, 2, [1])).toThrowError("length of data must exactly match size of matrix")
    })

    test("get and set data", () => {
        let a = new Matrix(3, 2)
        expect(a.data).toEqual([0, 0, 0, 0, 0, 0])
        const data = [1, 2, 3, 4, 5, 6]
        let index = 0
        for (let row = 0; row < a.rows; row++) {
            for (let col = 0; col < a.cols; col++) {
                a.setData(row, col, data[index])
                expect(a.getData(row, col)).toBe(data[index])
                index++
            }
        }
        expect(a.data).toEqual(data)

        expect(() => a.getData(4, 1)).toThrowError("row out of bounds")
        expect(() => a.getData(1, -2)).toThrowError("col out of bounds")
        expect(() => a.setData(-2, 1, 1)).toThrowError("row out of bounds")
        expect(() => a.setData(1, 100, -1)).toThrowError("col out of bounds")
    })

    test("self transpose", () => {
        let a = new Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
        a.transpose()
        expect(a.data).toEqual([1, 4, 7, 2, 5, 8, 3, 6, 9])
        expect(a.rows).toBe(3)
        expect(a.cols).toBe(3)

        a = new Matrix(2, 3, [1, 2, 3, 4, 5, 6])
        a.transpose()
        expect(a.data).toEqual([1, 4, 2, 5, 3, 6])
        expect(a.rows).toBe(3)
        expect(a.cols).toBe(2)
    })

    test("createTranspose", () => {
        let a = new Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
        let b = Matrix.createTranspose(a)
        expect(b.data).toEqual([1, 4, 7, 2, 5, 8, 3, 6, 9])
        expect(b.rows).toBe(3)
        expect(b.cols).toBe(3)

        a = new Matrix(2, 3, [1, 2, 3, 4, 5, 6])
        b = Matrix.createTranspose(a)
        expect(b.data).toEqual([1, 4, 2, 5, 3, 6])
        expect(b.rows).toBe(3)
        expect(b.cols).toBe(2)
    })

    test("matrix multiplication", () => {
        let a = new Matrix(2, 2, [1, 2, 3, 4])
        let b = new Matrix(2, 2, [4, 3, 2, 1])
        let c = Matrix.Multiply(a, b)
        expect(c.data).toEqual([8, 5, 20, 13])
        expect(c.rows).toBe(2)
        expect(c.cols).toBe(2)

        b = new Matrix(2, 3, [1, 2, 3, 4, 5, 6])
        c = Matrix.Multiply(a, b)
        expect(c.data).toEqual([9, 12, 15, 19, 26, 33])
        expect(() => Matrix.Multiply(b, a)).toThrowError("columns of first matrix must match rows of second matrix")
    })

    test("Matrix from Tensor", () => {
        let tensor = new Tensor([3, 3])
        let matrix = Matrix.fromTensor(tensor)
        expect(matrix.rows).toBe(3)
        expect(matrix.cols).toBe(3)
        expect(matrix.data).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0])

        tensor = new Tensor([2])
        matrix = Matrix.fromTensor(tensor)
        expect(matrix.rows).toBe(2)
        expect(matrix.cols).toBe(1)
        expect(matrix.data).toEqual([0, 0])

        tensor = new Tensor([3, 3, 3])
        expect(() => Matrix.fromTensor(tensor)).toThrowError("cardinality of tensor must be 1 or 2")
    })
})