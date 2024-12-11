import java.util.Random;

class Main {

    public static int[] sequenceGenerator(int size) {
        Random random_generator = new Random();
        int[] sequence = new int[size];

        for (int i = 0; i < size; i++) {
            sequence[i] = random_generator.nextInt(2);
        }

        return sequence;
    }


    public static void main(String[] args) {
        int[] result_sequence = sequenceGenerator(128);

        for (int i = 0; i < 128; i++) {
            System.out.print(result_sequence[i]);
        }
    }
}

// Performed on the website specified in the document for the 2nd laboratory

// 10011000001000011110001110011110001111010011100111100101000001001101111101111001010011101001110010011100110111100010100100110001