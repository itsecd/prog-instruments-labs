import java.util.Random;

/**
 * A class to generate random binary sequences.
 */
public class RandomBinarySequenceGenerator {

    /**
     * Generates a random binary sequence of the specified size.
     *
     * @param size The size of the sequence to generate.
     * @return A string representing the random binary sequence.
     */
    public static String generateRandomBinarySequence(int size) {
        Random random = new Random();
        StringBuilder binarySequence = new StringBuilder();

        // Generate random binary numbers and append them to the sequence.
        for (int i = 0; i < size; ++i) {
            int randomNumber = random.nextInt(2); // Generate a random number 0 or 1
            binarySequence.append(randomNumber); // Add the random number to the sequence
        }

        return binarySequence.toString();
    }

    /**
     * The main method of the program.
     *
     * @param args Command-line arguments (not used in this program).
     */
    public static void main(String[] args) {
        int size = 128; // The specified size of the sequence
        String randomBinarySequence = generateRandomBinarySequence(size);

        System.out.println("Random binary sequence: " + randomBinarySequence);
    }
}