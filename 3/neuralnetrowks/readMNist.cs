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

        private void setupReader(){
                brLabels = new BinaryReader(ifsLabels);
                brImages = new BinaryReader(ifsImages);

                
                int magic1 = brImages.ReadInt32(); // discard
                int numImages = brImages.ReadInt32();
                int numRows = brImages.ReadInt32();
                int numCols = brImages.ReadInt32();

                int magic2 = brLabels.ReadInt32();
                int numLabels = brLabels.ReadInt32();

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


