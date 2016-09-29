using System;
using System.Collections.Generic;
using System.IO;

public class inicio
{

#region "mierda"

    private static double letterToDouble(String p)
    {
        return Convert.ToUInt16(p.ToCharArray()[0]);
    }
    
    private static double[] stringTodouble(String p)
    {
        ushort[] usA = Array.ConvertAll(p.ToCharArray(), Convert.ToUInt16);
        double[] dRet = new double[usA.Length];

        for (Int32 i = 0; i < dRet.Length; i++)
            dRet[i] = usA[i];

        return dRet;
    }

    public static myType[][] shuffle<myType>(myType[][] array)
    {
        List<myType[]> lNew = new List<myType[]>(array);
        List<myType[]> lRet = new List<myType[]>();
        Random r = new Random();
        myType[] tmp;
        Int32 iTmp;

        while (lNew.Count != 0)
        {
            iTmp = r.Next(0, lNew.Count);
            tmp = lNew[iTmp];
            lRet.Add(tmp);
            lNew.Remove(tmp);
        }

        return lRet.ToArray();
    }

    public static T[][] arrayOfArray<T>(T[] p)
    {
        T[][] aRet = new T[p.Length][];
        for (Int32 i = 0; i < p.Length; i++)
            aRet[i] = new T[1] { p[i] };
        return aRet;
    }


    private static Int32 _getMaxIndexFromResults(double[] p){
        Int32 i, l = p.Length;
        Int32 index = -1;
        double max = double.MinValue;

        for(i = 0; i < l; i ++)
            if(p[i] > max){
                max = p[i];
                index = i;
            }

        return index;
    }

    public static bool pseudoTrainSetByEpocs(ref double[][] inputs, ref double[][] desired, NeuralNetwork n, 
                                Int32 epocs, string fileName, double lambdaRegParam, Int32 totalSamples, 
                                Int32 miniBatchSize, double learningRate)
    {
        Int32[][] indexes = new Int32[totalSamples][];
        Int32 i = 0;
        double[][] aInputs = new double[miniBatchSize][];
        double[][] aDesired = new double[miniBatchSize][];
        Int32 epocsI = 0;
        Int32 miniBatches = ((Int32)totalSamples) / miniBatchSize;
        Int32 miniBatchesI;
        double minMatches = 0, tmpMatches = 0;
        Int32 linePosition = 10;
        double tmpCost;
        //we set this to suffle indexes
        for(i = 0; i < totalSamples; i ++)
            indexes[i] = new Int32[1]{i};

        


        //now we are to train by epocs
        writeLog(fileName, String.Format("\"id\",\"epocs\", \"MiniBatch_Id\",\"test_matches\",\"test_cases\", \"lambda_Reg_\", \"Cost\",\"Cost_Function_Name\",\"Learning Rate\",\"Total Samples\""));

        for(epocsI = 0; epocsI < epocs; epocsI ++){
            //suffling inputs

           indexes = shuffle<Int32>(indexes);
            //now we train for every mini batch

            for(miniBatchesI = 0; miniBatchesI < miniBatches; miniBatchesI ++){
                for(i = 0; i < miniBatchSize; i ++){
                    aInputs[i] = inputs[indexes[miniBatchesI*miniBatchSize+i][0]];
                    aDesired[i] = desired[indexes[miniBatchesI*miniBatchSize+i][0]];
                }

                //we train it
                n.pseudoTrainSetSGD(aInputs, aDesired, lambdaRegParam, totalSamples);
                if(Console.KeyAvailable && Console.ReadKey(true).Key == ConsoleKey.Escape)
                    return false;

                Console.SetCursorPosition(0,linePosition); //,miniBatchSize+10);        
                Console.Write("Training minibatch {2} of {3} ON Epoc # {0} of {1})....                                       ",epocsI, epocs, miniBatchesI,  miniBatches);   
            }

            tmpMatches = testNetwork(n, 100);
            tmpCost = n.Cost.cost(aDesired, aInputs, n);
            writeLog(fileName, String.Format("{0},{1},{2},{3:N2},{4},{5},{6},\"{7}\",{8},{9}", epocsI, epocs, miniBatchesI, tmpMatches, 
                                            100, lambdaRegParam, tmpCost, n.Cost.getName(), learningRate, totalSamples));
            Console.SetCursorPosition(0,linePosition++);
            Console.WriteLine("Trained Epoc {0}, COST: {1:N3}, Accuracy: {2:N2}", epocsI, tmpCost, tmpMatches);
            
            n.saveToFile(fileName);
            if(tmpMatches >= minMatches ){
                minMatches = tmpMatches;
                n.saveToFile(fileName+".good.bin");
            }

        }
        return true;
    }


 private static Int32 guessDigit(double[] digit){

     double[] digitMatrix;
        //normaliza
        for(Int32 i = 0; i < 28; i ++)
            digit[i] = digit[i]>=0.5?1:0;

        Int32 number=-1;
        Int32 max=-1;
        Int32 counter = 0;
        for(Int32 i = 0; i < 10; i ++){
            counter = 0;
            digitMatrix = getDigits(i);
            for(Int32 j = 0; j < 28; j++)
                counter = counter + (digit[j]==digitMatrix[j]?1:0);
            if(counter > max){
                max = counter;
                number = i;
            }
        }

        return number;

    }


    private static double[] getDigits(Int32 i){
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

        return aRet[i];

    }


    private static void recognizeDigits(string fileName, Int32 miniBatchSize, Int32 epocs, double lambdaRegParam, 
                                        Int32 totalSamples, double learningRate, string costFunction)
    {
        NeuralNetwork myNet;
        readMNist digits = new readMNist("train-labels.idx1-ubyte", "train-images.idx3-ubyte");
        
        double[][] tryiningData = new double[totalSamples][];
        double[][] desired = new double[totalSamples][];
        Int32 tmp = 0;

        if(fileName.Length == 0 || (!File.Exists(fileName)))
            myNet = new NeuralNetwork( 784, 150,  10);
        else 
            myNet = NeuralNetwork.getFromFile(fileName);


        myNet.LearningRate = learningRate;

        switch (costFunction)
        {
            case "C":
            case "c":
                myNet.SetCostFunction(new CrossEntropy());
        
                break;
            case "Q":
            case "q":
                 myNet.SetCostFunction(new Quadratic());
                 break;
            

        }
        
        Console.Clear();

        //we get one sample 
        for(Int32 i = 0; i < totalSamples; i ++){
            digits.GiveNextValue(out tryiningData[i], ref tmp);
            //desired[i]= getDigits(tmp); 
            desired[i] = numpy.getArrayPopulated<double>(10,0);
            desired[i][tmp]=1;
        }

        pseudoTrainSetByEpocs(ref tryiningData, ref desired, myNet, epocs, fileName, lambdaRegParam, totalSamples, miniBatchSize, learningRate);
    

    }

    public static double testNetwork(NeuralNetwork myNet, Int32 testCases){
        double[] digit;
        double[] result;
        double max = 0.0;
        Int32 numb = 0;
        readMNist digits = new readMNist("t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte");
        Int32 tmp = 0;
        Int32 digitsMached = 0;

        for(Int32 j = 0; j < testCases; j++){    
            digits.GiveNextValue(out digit, ref tmp);
            result = myNet.Feedfordward(digit);
            max = -1.0;

            for(Int32 i = 0; i < result.Length; i++)
                if(result[i] > max){
                    max = result[i];
                    numb = i;
                }

//            numb = guessDigit(result);
            if(numb == tmp) digitsMached += 1;
        }

        digits.closeAll();
        
        return (((double)digitsMached)/((double)testCases))*100.0;
    }

    public static void writeLog(string fileName, string text){
        try{
        using (StreamWriter sw = File.AppendText(fileName+".csv")){
            sw.WriteLine(text);
        }
        }catch(Exception e){
            
            System.Threading.Thread.CurrentThread.Join(1000);
            writeLog(fileName, text);
        }
    }

  
#endregion

    public static void Main(string[] args)
    {

       // readMNist a = new readMNist("train-labels.idx1-ubyte", "train-images.idx3-ubyte");
        
        //param names: samples, epocs
        if(args.Length < 7){
            Console.WriteLine("\n\n=================\n Please use xxxxx.exe [file To Save] [MiniBatch size] [epocs] [lambda Reg.Param] [total Samples] [learning Rate] [Quad or Cross] \n=========\n");
            return;
        }
        recognizeDigits(args[0], Int32.Parse(args[1]), Int32.Parse(args[2]), double.Parse(args[3]), 
                    Int32.Parse(args[4]), double.Parse(args[5]), args[6] );

    }
}