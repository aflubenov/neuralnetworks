using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;



using System.IO;


    class readMNist
    {
        FileStream ifsLabels; 
        FileStream ifsImages;

        BinaryReader brLabels;
        BinaryReader brImages;

        public void reset(){
            brImages.BaseStream.Seek(0, SeekOrigin.Begin);
            brLabels.BaseStream.Seek(0, SeekOrigin.Begin);

            //we move file pointers
                brImages.ReadInt32(); // magic . discard
                brImages.ReadInt32(); // number of images
                brImages.ReadInt32(); //number of rows per image
                brImages.ReadInt32(); // number of columns per image

                brLabels.ReadInt32(); //other magic number
                brLabels.ReadInt32(); //number of labels
        }
        private void setupReader(){
                brLabels = new BinaryReader(ifsLabels);
                brImages = new BinaryReader(ifsImages);
                reset();
                

        }

        public void GiveNextValue(ref double[][] pixels, ref Int32 lbl ){
                    for (int i = 0; i < 28; ++i)
                    {
                        for (int j = 0; j < 28; ++j)
                        {
                            byte b = brImages.ReadByte();
                            pixels[i][j] = b;
                        }
                    }

                     lbl = brLabels.ReadByte();
                return;
        }

        public double[] GiveNextSpecificValue(Int32 p){
            double[] aRet = new double[28*28];
            Int32 iTmp = -1;

            for(;iTmp !=p;)
                GiveNextValue(out aRet, ref iTmp );

            return aRet;
        }

        public void GiveNextValue(out double[] pixels, ref Int32 lbl){
                pixels = new double[28*28];
                for (int i = 0; i < 28*28; ++i)
                    {
                        byte b = brImages.ReadByte();
                        pixels[i] = b;
                        
                    }

                     lbl = brLabels.ReadByte();
                return;
        }

        public readMNist(string labelsPath, string imagesPath)
        {
            //Console.WriteLine("\nBegin\n");
            ifsLabels = new FileStream(labelsPath, FileMode.Open); // test labels
            ifsImages = new FileStream(imagesPath, FileMode.Open); // test images

            setupReader();
            

            //Console.WriteLine("\nEnd\n");
            //Console.ReadLine();
        
        } // Main

        ~readMNist(){

            ifsImages.Close();
            brImages.Close();
            ifsLabels.Close();
            brLabels.Close();

        }
    } // Program


