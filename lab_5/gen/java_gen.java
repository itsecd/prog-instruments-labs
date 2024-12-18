package lab_2.gen;
import java.util.Random;

class Main {

    public static int[] generator(int size) {
        Random rand_generator = new Random();
        int[] rand_sequence = new int[size];

        for (int i = 0; i < size; i++) {
            rand_sequence[i] = rand_generator.nextInt(2);
        }

        return rand_sequence;
    }

    
    public static void main(String[] args) {
        int[] rand_sequence = generator(128);

        for (int i = 0; i < 128; i++) {
            System.out.print(rand_sequence[i]);
        }
    }
    
} 

// 01101000111000010001001110011001010100011110111000101010000011100001000110110111111000111001001110100111001110001101110010010000