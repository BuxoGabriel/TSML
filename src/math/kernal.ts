import Matrix from "./matrix";
import Tensor from "./tensor";

export default class Kernal extends Matrix {
    public bias: number
    public radius: number
    constructor(radius: number) {
        super(2 * radius + 1, 2 * radius + 1)
        this.radius = radius
        this.randomize(-1, 1, true)
        this.bias = Math.random() * 2 - 1
    }
    
    pass(images: Tensor, processedImages: Tensor, kernal: number): Tensor {
        if(images.cardinality != 3) throw Error("can only pass on 3d tensors")
        const [depth, rows, cols] = images.dimensions
        for(let i = 0; i < depth; i++) {
            for(let imageRow = 0; imageRow < rows; imageRow++) {
                for(let imageCol = 0; imageCol < cols; imageCol++) {
                    let index = kernal * images.size + i * rows * cols + imageRow * cols + imageCol
                    processedImages.data[index] = 0
                    for(let yoff = -this.radius; yoff <= this.radius; yoff++) {
                        for(let xoff = -this.radius; xoff <= this.radius; xoff++) {
                            let y = imageRow + yoff
                            let x = imageCol + xoff
                            if(y < 0 || y >= rows) continue
                            if(x < 0 || x >= cols) continue
                            processedImages.data[index] += this.getData(yoff + this.radius, xoff + this.radius) * images.data[i * rows * cols + y * cols + x]
                        }
                    }
                    processedImages.data[index] += this.bias
                }
            }
        }
        return processedImages
    }

    add(other: Kernal, inPlace = false): Kernal {
        let kernal: Kernal
        if(inPlace) kernal = this
        else kernal = new Kernal(this.radius)
        
        for(let i = 0; i < this.size; i++) {
            kernal.data[i] = this.data[i] + other.data[i]
        }
        kernal.bias = this.bias + other.bias

        return kernal
    }

    zero(): Kernal {
        this.data = this.data.map(x => 0)
        this.bias = 0
        return this
    }
}