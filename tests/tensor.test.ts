import { describe, expect, test } from "@jest/globals"
import Tensor from "../src/math/tensor"

describe('tensor module', () => {
    test('tensor initialization', () => {
        let tensor = new Tensor([1])
        expect(tensor.data.length).toBe(1)
        expect(tensor.size).toBe(1)
        expect(tensor.cardinality).toBe(1)
        expect(tensor.data).toEqual([0])

        tensor = new Tensor([5, 5, 5])
        expect(tensor.data.length).toBe(125)
        expect(tensor.size).toBe(125)
        expect(tensor.cardinality).toBe(3)

        expect(() => new Tensor([])).toThrowError("tensor can not be 0 dimensional")
        expect(() => new Tensor([3, 2, 0])).toThrowError("tensor can not have dimension with size less than 1")
        expect(() => new Tensor([-2, 2, 2])).toThrowError("tensor can not have dimension with size less than 1")
        expect(() => new Tensor([2, 2], [])).toThrowError("length of data must exactly match size of tensor")

        tensor = new Tensor([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8])

    })

    test('tensor signatures', () => {
        let t1 = new Tensor([5, 6])
        let t2 = new Tensor([5, 6])
        expect(t1.matchesSignature(t2)).toBeTruthy()
        t2 = new Tensor([1, 1])
        expect(t1.matchesSignature(t2)).toBeFalsy()
        t2 = new Tensor([3, 10, 1])
        expect(t1.matchesSignature(t2)).toBeFalsy()
    })

    describe('tensor operations', () => {
        test('generic operation', () => {
            let t1 = new Tensor([2, 2], [1, 2, 3, 4])
            let t2 = new Tensor([2, 2], [1, 1, 1, 1])

            let t3 = t1.operation(t2, (a, b) => b)
            expect(t3).not.toBe(t1)
            expect(t3.data).toEqual(t2.data)

            t3 = t1.operation(t2, (a, b) => a + b, true)
            expect(t3).toBe(t1)
            expect(t1.data).toEqual([2, 3, 4, 5])

            t2 = new Tensor([2, 1], [1, 1])
            expect(() => t1.operation(t2, (a, b) => b)).toThrowError("dimensions of tensors must match for subtraction")
        })

        test('tensor addition', () => {
            let t1 = new Tensor([2, 2], [1, 2, 3, 4])
            let t2 = new Tensor([2, 2], [1, 1, 1, 1])

            let t3 = t1.add(t2)
            expect(t3).not.toBe(t1)
            expect(t3.data).toEqual([2, 3, 4, 5])

            t3 = t1.add(t2, true)
            expect(t3).toBe(t1)
            expect(t1.data).toEqual([2, 3, 4, 5])
        })

        test('tensor subtraction', () => {
            let t1 = new Tensor([2, 2], [1, 2, 3, 4])
            let t2 = new Tensor([2, 2], [1, 1, 1, 1])

            let t3 = t1.subtract(t2)
            expect(t3).not.toBe(t1)
            expect(t3.data).toEqual([0, 1, 2, 3])

            t3 = t1.subtract(t2, true)
            expect(t3).toBe(t1)
            expect(t1.data).toEqual([0, 1, 2, 3])
        })

        test('tensor piecewise multiplication', () => {
            let t1 = new Tensor([2, 2], [1, 2, 3, 4])
            let t2 = new Tensor([2, 2], [2, 2, 2, 2])

            let t3 = t1.piecewiseMultiply(t2)
            expect(t3).not.toBe(t1)
            expect(t3.data).toEqual([2, 4, 6, 8])

            t3 = t1.piecewiseMultiply(t2, true)
            expect(t3).toBe(t1)
            expect(t1.data).toEqual([2, 4, 6, 8])
        })
    })

    test("tensor cloning", () => {
        let t1 = new Tensor([2, 2], [1, 2, 3, 4])
        let t2 = Tensor.clone(t1)
        expect(t1).not.toBe(t2)
        expect(t1.data).toEqual(t2.data)
    })
})