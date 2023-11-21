export type activationFunction =  {
    fn: (x: number) => number,
    dfn: (x: number) => number
}

export const sigmoid: activationFunction = {
    fn: (x) => 1/(1 + Math.exp(-x)),
    dfn: (x) => x * (1 - x)
}

export const relu: activationFunction = {
    fn: (x) => Math.max(0, x),
    dfn: (x) => x > 0? 1 : 0
}

export type lossFunction = {
    fn: (expected: number, result: number) => number,
    // expected is treated as a constant and derivative is with respect to result
    dfn: (expected: number, result: number) => number
}

export const squareErr: lossFunction = {
    fn: (exp, res) => (exp - res) * (exp - res) / 2,
    dfn: (exp, res) => res - exp
}