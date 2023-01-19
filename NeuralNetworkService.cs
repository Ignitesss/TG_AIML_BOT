using System.Drawing;
using System.Linq;
using AForge.Imaging.Filters;

namespace AIMLTGBot
{
    public class NeuralNetworkService
    {
        private StudentNetwork net;
        public double[] processImage(Bitmap original)
        {
            // На вход поступает необработанное изображение с веб-камеры

            int xborder = original.Width / 10, yborder = original.Height / 10;
            //  Мы сейчас занимаемся тем, что красиво оформляем входной кадр, чтобы вывести его на форму
            //Rectangle cropRect = new Rectangle((bitmap.Width - bitmap.Height) / 2 + settings.left + settings.border, settings.top + settings.border, side, side);
            Rectangle cropRect = new Rectangle(xborder, yborder, original.Width - xborder, original.Height - yborder);

            //  Объект для рисования создаём
            //Graphics g = Graphics.FromImage(original);

            //g.DrawImage(bitmap, new Rectangle(0, 0, original.Width, original.Height), cropRect, GraphicsUnit.Pixel);

            //  Теперь всю эту муть пилим в обработанное изображение
            AForge.Imaging.Filters.Crop cropFilter = new AForge.Imaging.Filters.Crop(cropRect);
            var uProcessed = cropFilter.Apply(AForge.Imaging.UnmanagedImage.FromManagedImage(original));
            AForge.Imaging.Filters.Grayscale grayFilter = new AForge.Imaging.Filters.Grayscale(0.2125, 0.7154, 0.0721);
            uProcessed = grayFilter.Apply(uProcessed);



            //  Масштабируем изображение до 500x500 - этого достаточно
            //AForge.Imaging.Filters.ResizeBilinear scaleFilter = new AForge.Imaging.Filters.ResizeBilinear(300,300);
            //uProcessed = scaleFilter.Apply(uProcessed);
            //original = scaleFilter.Apply(original);

            //  Пороговый фильтр применяем. Величина порога берётся из настроек, и меняется на форме
            AForge.Imaging.Filters.BradleyLocalThresholding threshldFilter = new AForge.Imaging.Filters.BradleyLocalThresholding();
            threshldFilter.PixelBrightnessDifferenceLimit = 0.13f;
            threshldFilter.ApplyInPlace(uProcessed);
            
            AForge.Imaging.BlobCounter blobber = new AForge.Imaging.BlobCounter();
            blobber.MinHeight = 5;
            blobber.MinWidth = 5;
            blobber.ObjectsOrder = AForge.Imaging.ObjectsOrder.XY;

            AForge.Imaging.Filters.Invert InvertFilter = new AForge.Imaging.Filters.Invert();
            InvertFilter.ApplyInPlace(uProcessed);
            
            blobber.ProcessImage(uProcessed);
            var rects = blobber.GetObjectsRectangles().Where(x=> x.Width > 5 && x.Height > 5);
            double scaleFactor = rects.Max(x => x.Width);
            var res = rects.Take(5).Select(x => x.Width / scaleFactor).Where(x=> x > 0.1).ToList();
            while (res.Count < 5)
            {
                res.Add(0);
            }

            return res.Select(x=>x * scaleFactor).ToArray();
        }

        public NeuralNetworkService()
        {
            net = StudentNetwork.readFromFile("net10.dat");
        }

        public string predict(Bitmap bitmap)
        {
            Sample sample = new Sample(processImage(bitmap), 10);
            return sample.ProcessPrediction(net.Compute(sample.input));
        }
    }
}