import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class CNN {
    private val interpreter: Interpreter

    init {
        // Load the TFLite model
        val tfliteModel: MappedByteBuffer = loadModelFile()
        interpreter = Interpreter(tfliteModel)
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = FileInputStream("model.tflite").fd
        val fileChannel = FileInputStream("model.tflite").channel
        val startOffset = fileChannel.position()
        val declaredLength = fileDescriptor.length - startOffset
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun predict(input: FloatArray): FloatArray {
        // Run inference on the input
        val output = Array(1) { FloatArray(10) }
        interpreter.run(input, output)
        return output[0]
    }
}
