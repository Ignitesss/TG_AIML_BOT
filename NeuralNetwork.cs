using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ProtoBuf;

namespace AIMLTGBot
{
    [ProtoContract]
    class Neuron
    {
        public static Func<double, double> activationFunction;
        public static Func<double, double> activationFunctionDerivative;
        [ProtoMember(1)]
        public int id;
        [ProtoMember(2)]
        public double Output;
        [ProtoMember(3)]
        public int layer;
        [ProtoMember(4)]
        public double error;
        [ProtoMember(5)]
        public double[] weightsToPrevLayer;

        public void setInput(double input)
        {
            if (layer == 0)
            {
                Output = input;
                return;
            }

            Output = activationFunction(input);
        }

        public Neuron(int id, int layer, int prevLayerCapacity, Random random)
        {
            this.id = id;
            this.layer = layer;
            this.error = 0;
            
            if (layer == -1)
            {
                Output = 1;
            }
            
            if (layer < 1)
            {
                weightsToPrevLayer = null;
            }
            else
            {
                weightsToPrevLayer = new double [prevLayerCapacity + 1];
                for (int i = 0; i < weightsToPrevLayer.Length; i++)
                {
                    weightsToPrevLayer[i] = random.NextDouble() * 2 - 1;
                }
            }
        }

        public Neuron()
        {
        }
    }

    [ProtoContract]
    class Layer
    {
        [ProtoMember(1)]
        public Neuron[] neurons;

        public Neuron this[int index]
        {
            get => neurons[index];
            set => neurons[index] = value;
        }

        public Layer(int capacity)
        {
            neurons = new Neuron[capacity];
        }

        public Layer()
        {
        }

        public int Length => neurons.Length;
        public IEnumerable<Neuron> Select(Func<Neuron, Neuron> selector) => neurons.Select(selector);
        public IEnumerable<double> Select(Func<Neuron, double> selector) => neurons.Select(selector);
    }

    [ProtoContract]
    public class StudentNetwork
    {
        [ProtoMember(1)]
        private const double learningRate = 0.1;
        [ProtoMember(2)]
        private Neuron biasNeuron;
        [ProtoMember(3)]
        private List<Layer> layers;
        [ProtoMember(4)]
        public Func<double[], double[], double> lossFunction;
        [ProtoMember(5)]
        public Func<double, double, double> lossFunctionDerivative;

        public StudentNetwork()
        {
        }

        public static StudentNetwork readFromFile(string filename)
        {
            Neuron.activationFunction = s => 1.0 / (1.0 + Math.Exp(-s));
            Neuron.activationFunctionDerivative = s => s * (1 - s);
            StudentNetwork network;
            using(MemoryStream memStream = new MemoryStream())
            {
                var arrBytes = File.ReadAllBytes(filename);
                memStream.Write(arrBytes, 0, arrBytes.Length);
                memStream.Seek(0, SeekOrigin.Begin);
                network = Serializer.Deserialize<StudentNetwork>(memStream);
            }
            return network;
        }
        public StudentNetwork(int[] structure)
        {
            if (structure.Length < 3)
            {
                throw new ArgumentException("Сетка из 0 слоёв это круто, но не пойдёт");
            }

            lossFunction = (output, aim) =>
            {
                double res = 0;
                for (int i = 0; i < aim.Length; i++)
                {
                    res += Math.Pow(aim[i] - output[i], 2);
                }

                return res * 0.5;
            };

            lossFunctionDerivative = (output, aim) => aim - output;

            Neuron.activationFunction = s => 1.0 / (1.0 + Math.Exp(-s));
            Neuron.activationFunctionDerivative = s => s * (1 - s);

            Random random = new Random();

            biasNeuron = new Neuron(0, -1, -1, random);
            int id = 1;

            layers = new List<Layer>();

            for (int layer = 0; layer < structure.Length; layer++)
            {
                layers.Add(new Layer(structure[layer]));
                for (int i = 0; i < structure[layer]; i++)
                {
                    if (layer == 0)
                    {
                        layers[layer][i] = new Neuron(id, layer, -1, random);
                        continue;
                    }

                    layers[layer][i] = new Neuron(id, layer, structure[layer - 1], random);

                    id++;
                }
            }
        }


        public void forwardPropagation(double[] input)
        {
            if (input.Length != layers[0].Length)
            {
                throw new ArgumentException("Вы мне подсунули какой-то странный входной массив.");
            }

            for (int i = 0; i < layers[0].Length; i++)
            {
                layers[0][i].setInput(input[i]);
            }

            for (int layer = 1; layer < layers.Count; layer++)
            {
                for (int neuron = 0; neuron < layers[layer].Length; neuron++)
                {
                    double scalar = 0;

                    for (int i = 0; i < layers[layer][neuron].weightsToPrevLayer.Length; i++)
                    {
                        if (i == 0)
                        {
                            scalar += biasNeuron.Output * layers[layer][neuron].weightsToPrevLayer[0];
                            continue;
                        }

                        scalar += layers[layer - 1][i - 1].Output * layers[layer][neuron].weightsToPrevLayer[i];
                    }

                    layers[layer][neuron].setInput(scalar);
                }
            }
        }

        public void backwardPropagation(Sample sample)
        {
            var aim = sample.outputVector;

            for (var i = 0; i < layers.Last().Length; i++)
            {
                layers.Last()[i].error = lossFunctionDerivative(layers.Last()[i].Output, aim[i]);
            }

            for (int layer = layers.Count - 1; layer >= 1; layer--)
            {
                for (int k = 0; k < layers[layer].Length; k++)
                {
                    var neuron = layers[layer][k];

                    neuron.error *= Neuron.activationFunctionDerivative(neuron.Output);

                    for (int i = 0; i < neuron.weightsToPrevLayer.Length; i++)
                    {
                        if (i == 0)
                        {
                            biasNeuron.error += neuron.error * neuron.weightsToPrevLayer[0];
                            neuron.weightsToPrevLayer[0] += learningRate * neuron.error * biasNeuron.Output;
                            continue;
                        }

                        layers[layer - 1][i - 1].error += neuron.error * neuron.weightsToPrevLayer[i];
                        neuron.weightsToPrevLayer[i] += learningRate * neuron.error * layers[layer - 1][i - 1].Output;
                    }

                    neuron.error = 0;
                }
            }
        }

        double TrainOnSample(Sample sample, double acceptableError)
        {
            double loss;
            forwardPropagation(sample.input);
            loss = lossFunction(layers.Last().Select(n => n.Output).ToArray(), sample.outputVector);
            backwardPropagation(sample);
            return loss;
        }        

        public double[] Compute(double[] input)
        {
            if (input.Length != layers[0].Length)
            {
                throw new ArgumentException("У вас тут данных многовато...");
            }

            forwardPropagation(input);
            return layers.Last().Select(n => n.Output).ToArray();
        }

        public void save(string filename)
        {
            using (MemoryStream ms = new MemoryStream())
            {
                Serializer.Serialize(ms, this);
                File.WriteAllBytes(filename,ms.ToArray());
            }
        }
    }
    public enum LetterType : byte { SH = 0, N, G, E, P, T, TS, Z, A, SOFT, Undef };
    public class Sample
    {
        /// <summary>
        /// Входной вектор
        /// </summary>
        public double[] input = null;

        /// <summary>
        /// Вектор ошибки, вычисляется по какой-нибудь хитрой формуле
        /// </summary>
        public double[] error = null;

        /// <summary>
        /// Действительный класс образа. Указывается учителем
        /// </summary>
        public LetterType actualClass;

        /// <summary>
        /// Распознанный класс - определяется после обработки
        /// </summary>
        public LetterType recognizedClass;

        public double[] outputVector;

        /// <summary>
        /// Конструктор образа - на основе входных данных для сенсоров, при этом можно указать класс образа, или не указывать
        /// </summary>
        /// <param name="inputValues"></param>
        /// <param name="sampleClass"></param>
        public Sample(double[] inputValues, int classesCount, LetterType sampleClass = LetterType.Undef)
        {
            //  Клонируем массивчик
            input = (double[]) inputValues.Clone();
            Output = new double[classesCount];
            if (sampleClass != LetterType.Undef) Output[(int) sampleClass] = 1;


            recognizedClass = LetterType.Undef;
            actualClass = sampleClass;
            
            outputVector = new double[classesCount];
            for (int i = 0; i < outputVector.Length; i++)
            {
                outputVector[i] = i == (int)actualClass ? 1: 0;
            }
        }

        /// <summary>
        /// Выходной вектор, задаётся извне как результат распознавания
        /// </summary>
        public double[] Output { get; private set; }

        public static string LetterTypeToString(LetterType type)
        {
            switch (type)
            {
                case LetterType.SH:
                    return "Ш";
                case LetterType.N:
                    return "Н";
                case LetterType.G:
                    return "Г";
                case LetterType.E:
                    return "Е";
                case LetterType.SOFT:
                    return "Ь";
                case LetterType.Z:
                    return "З";
                case LetterType.T:
                    return "Т";
                case LetterType.TS:
                    return "Ц";
                case LetterType.P:
                    return "П";
                case LetterType.A:
                    return "А";
                case LetterType.Undef:
                    return "Неизвестно";
                default:
                    throw new ArgumentOutOfRangeException(nameof(type), type, null);
            }
        }
        /// <summary>
        /// Обработка реакции сети на данный образ на основе вектора выходов сети
        /// </summary>
        public string ProcessPrediction(double[] neuralOutput)
        {
            Output = neuralOutput;
            if (error == null)
                error = new double[Output.Length];

            //  Нам так-то выход не нужен, нужна ошибка и определённый класс
            recognizedClass = 0;
            for (int i = 0; i < Output.Length; ++i)
            {
                error[i] = (Output[i] - (i == (int) actualClass ? 1 : 0));
                if (Output[i] > Output[(int) recognizedClass]) recognizedClass = (LetterType) i;
            }

            return LetterTypeToString(recognizedClass);
        }

        /// <summary>
        /// Вычисленная суммарная квадратичная ошибка сети. Предполагается, что целевые выходы - 1 для верного, и 0 для остальных
        /// </summary>
        /// <returns></returns>
        public double EstimatedError()
        {
            double Result = 0;
            for (int i = 0; i < Output.Length; ++i)
                Result += Math.Pow(error[i], 2);
            return Result;
        }

        /// <summary>
        /// Добавляет к аргументу ошибку, соответствующую данному образу (не квадратичную!!!)
        /// </summary>
        /// <param name="errorVector"></param>
        /// <returns></returns>
        public void updateErrorVector(double[] errorVector)
        {
            for (int i = 0; i < errorVector.Length; ++i)
                errorVector[i] += error[i];
        }

        /// <summary>
        /// Представление в виде строки
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            string result = "Sample decoding : " + actualClass.ToString() + "(" + ((int) actualClass).ToString() +
                            "); " + Environment.NewLine + "Input : ";
            for (int i = 0; i < input.Length; ++i) result += input[i].ToString() + "; ";
            result += Environment.NewLine + "Output : ";
            if (Output == null) result += "null;";
            else
                for (int i = 0; i < Output.Length; ++i)
                    result += Output[i].ToString() + "; ";
            result += Environment.NewLine + "Error : ";

            if (error == null) result += "null;";
            else
                for (int i = 0; i < error.Length; ++i)
                    result += error[i].ToString() + "; ";
            result += Environment.NewLine + "Recognized : " + recognizedClass.ToString() + "(" +
                      ((int) recognizedClass).ToString() + "); " + Environment.NewLine;


            return result;
        }

        /// <summary>
        /// Правильно ли распознан образ
        /// </summary>
        /// <returns></returns>
        public bool Correct()
        {
            return actualClass == recognizedClass;
        }
    }
}