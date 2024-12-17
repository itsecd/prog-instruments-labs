import java.util.Random;
import java.util.BitSet;

public class Main {
    public static BitSet generateRandomSequence() {
        Random generator = new Random();
        BitSet bitSet = new BitSet(128);
        for (int i = 0; i < 128; i++) {
            if (generator.nextBoolean()) {
                bitSet.set(i);
            }
        }
        return bitSet;
    }

    public static void main(String[] args)
    {
        BitSet bi = generateRandomSequence();
        StringBuilder s = new StringBuilder();
        for( int i = 0; i < bi.length();  i++ )
        {
            s.append( bi.get( i ) == true ? 1: 0 );
        }

        System.out.println( s );
    }
}
