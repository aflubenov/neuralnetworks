using System;
using System.Collections.Generic;
using System.IO;
using System.Drawing;

public class inicio
{

#region "mierda"

    public static T[][] arrayOfArray<T>(T[] p)
    {
        T[][] aRet = new T[p.Length][];
        for (Int32 i = 0; i < p.Length; i++)
            aRet[i] = new T[1] { p[i] };
        return aRet;
    }


    private static double[] imgToDouble(string filename){
        Bitmap image = new Bitmap( Image.FromFile(filename, true));
        double[] aRet = new double[28*28];

        //we know image is 28x28
        for(Int32 i = 0; i < 28; i ++){
            Console.Write("\n");
            for(Int32 j= 0; j < 28; j++){
                aRet[28*i+j] =255.0 - image.GetPixel(j,i).R;
                Console.Write("{0} ",aRet[28*i+j] > 50?"*":" ");
            }
        }
        
        return aRet;


    }

    private static Int32 guessDigit(double[] digit){
        double[][] aRet = new double[10][];

        aRet[0] = new double[28]{0,1,1,0,
                                 1,0,0,1,
                                 1,0,0,1,
                                 1,0,0,1,
                                 1,0,0,1,
                                 1,0,0,1,
                                 0,1,1,0,
                                 };
                                
        aRet[1] = new double[28]{0,0,1,0,
                                 0,1,1,0,
                                 1,0,1,0,
                                 0,0,1,0,
                                 0,0,1,0,
                                 0,0,1,0,
                                 0,0,1,0,
                                 };

        aRet[2] = new double[28]{0,1,1,1,
                                 1,0,0,1,
                                 0,0,0,1,
                                 0,0,1,0,
                                 0,1,0,0,
                                 1,0,0,0,
                                 1,1,1,1,
                                 };

        aRet[3] = new double[28]{0,1,1,0,
                                 1,0,0,1,
                                 0,0,0,1,
                                 0,1,1,0,
                                 0,0,0,1,
                                 1,0,0,1,
                                 0,1,1,0,
                                 };
        aRet[4] = new double[28]{1,0,0,1,
                                 1,0,0,1,
                                 1,1,1,1,
                                 0,0,0,1,
                                 0,0,0,1,
                                 0,0,0,1,
                                 0,0,0,1,
                                 };

        aRet[5] = new double[28]{1,1,1,1,
                                 1,0,0,0,
                                 1,1,1,0,
                                 0,0,0,1,
                                 0,0,0,1,
                                 1,0,0,1,
                                 0,1,1,0,
                                 };

        aRet[6] = new double[28]{0,1,1,0,
                                 1,0,0,0,
                                 1,0,0,0,
                                 1,1,1,0,
                                 1,0,0,1,
                                 1,0,0,1,
                                 0,1,1,0,
                                 };
        aRet[7] = new double[28]{0,1,1,1,
                                 1,0,0,1,
                                 0,0,1,0,
                                 0,0,1,0,
                                 0,0,1,0,
                                 0,1,0,0,
                                 0,1,0,0,
                                 };

        aRet[8] = new double[28]{0,1,1,0,
                                 1,0,0,1,
                                 1,0,0,1,
                                 0,1,1,0,
                                 1,0,0,1,
                                 1,0,0,1,
                                 0,1,1,0,
                                 };

        aRet[9] = new double[28]{0,1,1,0,
                                 1,0,0,1,
                                 1,0,0,1,
                                 0,1,1,1,
                                 0,0,0,1,
                                 0,0,0,1,
                                 0,1,1,0,
                                 };

        //normaliza
        for(Int32 i = 0; i < 28; i ++)
            digit[i] = digit[i]>=0.5?1:0;

        Int32 number=-1;
        Int32 max=-1;
        Int32 counter = 0;
        for(Int32 i = 0; i < 10; i ++){
            counter = 0;
            for(Int32 j = 0; j < 28; j++)
                counter = counter + (digit[j]==aRet[i][j]?1:0);
            if(counter > max){
                max = counter;
                number = i;
            }
        }

        return number;

    }


    private static void recognizeDigits(string fileName, string digitImageBitmap)
    {
        NeuralNetwork myNet;
        double[] digit;
        double[] result;
        double max = 0.0;
        Int32 numb = 0;
        readMNist digits = new readMNist("t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte");
        Int32 tmp = 0;
        double digitsReaded = 0.0, digitsMached = 0.0;

        //digit = imgToDouble(digitImageBitmap);
     
        myNet = NeuralNetwork.getFromFile(fileName); // new NeuralNetwork( 784, 15, 10 );

        for(;;){    
       digits.GiveNextValue(out digit, ref tmp);
        result = myNet.Feedfordward(digit);
        
        for(Int32 i = 0; i < 28; i ++){
            Console.Write("\n");
            for(Int32 j = 0; j < 28; j ++)
                Console.Write("{0}", digit[i*28+j] > 0?"*":" ");
        }

         Console.Write("\n");
          Console.Write("\n");
        for(Int32 i = 0; i < 28; i ++){
            if(i % 4 == 0) Console.Write("\n");
            Console.Write(result[i] >=0.5?"*":" ");
        }

        max = -1.0;

/*        for(Int32 i = 0; i < result.Length; i++)
            if(result[i] > max){
                max = result[i];
                numb = i;
            }
        */
        numb = guessDigit(result);        

        digitsReaded +=1.0;
        if(numb == tmp) digitsMached += 1.0;

        Console.Write(" Es un {0} [original: {4}] ????     Aciertos: {1:N0} de {2:N0}= {3:N2}%", numb, digitsMached, digitsReaded, (digitsMached/digitsReaded)*100.0, tmp );

        Console.WriteLine("\n");
        Console.ReadLine();
    }
    }


#endregion

    public static void Main(string[] args)
    {

        // readMNist a = new readMNist("train-labels.idx1-ubyte", "train-images.idx3-ubyte");

        //recognizeOneLetter();

        recognizeDigits("recognizeHandWritedDigits.bin", "x.x"); // args[0]);

    }
}