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

    private static void recognizeDigits(string fileName, string digitImageBitmap)
    {
        NeuralNetwork myNet;
        double[] digit;
        double[] result;
        double max = 0.0;
        double numb = 0;

        digit = imgToDouble(digitImageBitmap);

        myNet = NeuralNetwork.getFromFile(fileName); // new NeuralNetwork( 784, 15, 10 );

        

        result = myNet.Feedfordward(digit);
        
        for(Int32 i = 0; i < result.Length; i++)
            if(result[i] > max){
                max = result[i];
                numb = i;
            }
        
            Console.Write(" Es un {0} ???? ", numb);

        Console.WriteLine("\n");
    }


#endregion

    public static void Main(string[] args)
    {

       // readMNist a = new readMNist("train-labels.idx1-ubyte", "train-images.idx3-ubyte");
        
        //recognizeOneLetter();

        //recognizeDigits("recognizeHandWritedDigits.bin", args[0]);

        double[][] a = new double[3][]; //3 columns, 4 rows
        double[][] b = new double[2][]; //2 rows, 4 columns
        double[][] c = new double[2][]; //2 rows, 3 columns  

        a[0] = new double[4]{1,2,3,4};
        a[1] = new double[4]{5,6,7,8};
        a[2] = new double[4]{9,8,7,6};
        
        b[0] = new double[4]{1,2,3,4};
        b[1] = new double[4]{5,6,7,8};

        c[0] = new double[3];
        c[1] = new double[3];

        
        numpy.matrixMult(a,b,ref c);

        Console.Write("\n\n\n");
        for(Int32 i= 0; i < 2; i ++){
            for(Int32 j = 0; j < 3; j ++ )
                Console.Write("[{0}]  ", c[i][j]);
            Console.Write("\n");
        }

        Console.Write("\n\n\n");
    }
}