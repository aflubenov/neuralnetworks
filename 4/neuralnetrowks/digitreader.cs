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
        max = -1.0;

        for(Int32 i = 0; i < result.Length; i++)
            if(result[i] > max){
                max = result[i];
                numb = i;
            }
        
        digitsReaded +=1.0;
        if(numb == tmp) digitsMached += 1.0;
          
        Console.Write(" Es un {0} ????     Aciertos: {1:N0} de {2:N0}= {3:N2}%", numb, digitsMached, digitsReaded, (digitsMached/digitsReaded)*100.0 );

        Console.WriteLine("\n");
        Console.ReadLine();
    }
    }


#endregion

    public static void Main(string[] args)
    {

       // readMNist a = new readMNist("train-labels.idx1-ubyte", "train-images.idx3-ubyte");
        
        //recognizeOneLetter();

        recognizeDigits("recognizeHandWritedDigits.bin", args[0]);

    }
}