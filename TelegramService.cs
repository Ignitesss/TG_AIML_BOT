using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Threading;
using System.Threading.Tasks;
using Telegram.Bot;
using Telegram.Bot.Exceptions;
using Telegram.Bot.Extensions.Polling;
using Telegram.Bot.Types;
using Telegram.Bot.Types.Enums;

namespace AIMLTGBot
{
    enum ChatMode
    {
        CHATTING,
        RECOGNIZING
    };

    public class TelegramService : IDisposable
    {
        private readonly TelegramBotClient client;
        private readonly AIMLService aimlService;
        private readonly NeuralNetworkService networkService;
        private string lastRecognizedLetter = "none";

        Dictionary<long,ChatMode> dialogMode;
        private readonly CancellationTokenSource cts = new CancellationTokenSource();
        public string Username { get; }

        public TelegramService(string token, AIMLService aimlService)
        {
            this.aimlService = aimlService;
            networkService = new NeuralNetworkService();
            client = new TelegramBotClient(token);
            dialogMode = new Dictionary<long, ChatMode>();
            
            client.StartReceiving(HandleUpdateMessageAsync, HandleErrorAsync, new ReceiverOptions
            { 
                AllowedUpdates = new[] { UpdateType.Message }
            },
            cancellationToken: cts.Token);

            Username = client.GetMeAsync().Result.Username;
        }

        async Task HandleUpdateMessageAsync(ITelegramBotClient botClient, Update update, CancellationToken cancellationToken)
        {
            var message = update.Message;
            var chatId = message.Chat.Id;
            if (!dialogMode.ContainsKey(chatId))
            {
                dialogMode.Add(chatId,ChatMode.CHATTING);
            }
            var username = message.Chat.FirstName;
            if (message.Type == MessageType.Text)
            {
                var messageText = update.Message.Text.Replace("ё","е");
                Console.WriteLine($"Received a '{messageText}' message in chat {chatId} with {username}.");

                if (messageText == "/morse")
                {
                    lastRecognizedLetter = "none";
                    dialogMode[chatId] = ChatMode.RECOGNIZING;
                    await botClient.SendTextMessageAsync(
                        chatId: chatId,
                        text: "Скинь фоточку одной из этих букв (азбукой Морзе, само собой): П, Т, Ц, А, Г, Е, З, Н, Ш, Ь.\nМоя точность распознавания ококо 90%, не обижайся, если не смогу распознать",
                        cancellationToken: cancellationToken);
                    return;
                }
                
                if (dialogMode[chatId] == ChatMode.CHATTING)
                {
                    await botClient.SendTextMessageAsync(
                        chatId: chatId,
                        text: aimlService.Talk(chatId, username, messageText),
                        cancellationToken: cancellationToken);
                    return;
                }

                if (dialogMode[chatId] == ChatMode.RECOGNIZING)
                {
                    if (messageText.ToLower() == "хватит")
                    {
                        lastRecognizedLetter = "none";
                        await botClient.SendTextMessageAsync(
                            chatId: chatId,
                            text: "Окей, тогда давай болтать!",
                            cancellationToken: cancellationToken);
                        dialogMode[chatId] = ChatMode.CHATTING;
                        return;
                    }
                    if (lastRecognizedLetter!="none")
                    {
                        string answer;
                        if (messageText.Length > 1)
                        {
                            answer = "Вот так вот, ждешь букву, а тебе суют это...";
                        }
                        else if (messageText.ToUpper() == lastRecognizedLetter)
                        {
                            answer = aimlService.Talk(chatId, username, "угадал");
                        }
                        else
                        {
                            answer = aimlService.Talk(chatId, username, "промахнулся");
                            
                        }
                        await botClient.SendTextMessageAsync(
                            chatId: chatId,
                            text: answer,
                            cancellationToken: cancellationToken);
                        await botClient.SendTextMessageAsync(
                            chatId: chatId,
                            text: "Если хочешь болтать дальше, напиши \"хватит\", ну или продолжим играть в угадайку",
                            cancellationToken: cancellationToken);
                        
                        lastRecognizedLetter = "none";
                    }
                    else
                    {
                        await botClient.SendTextMessageAsync(
                            chatId: chatId,
                            text: "Мы тут болтать будем, или всё-таки морзе распознавать?",
                            cancellationToken: cancellationToken);
                    }
                }
            }

            // Загрузка изображений пригодится для соединения с нейросетью
            if (message.Type == MessageType.Photo)
            {
                if (dialogMode[chatId] != ChatMode.RECOGNIZING)
                {
                    await botClient.SendTextMessageAsync(
                        chatId: chatId,
                        text: "Фотка красивая, но мы же сейчас не в распознавание букв играем, не?",
                        cancellationToken: cancellationToken);
                    return;
                }
                var photoId = message.Photo.Last().FileId;
                Telegram.Bot.Types.File fl = await client.GetFileAsync(photoId, cancellationToken: cancellationToken);
                var imageStream = new MemoryStream();
                await client.DownloadFileAsync(fl.FilePath, imageStream, cancellationToken: cancellationToken);
                var img = new Bitmap(Image.FromStream(imageStream));
                var predicted = networkService.predict(img);
                
                await botClient.SendTextMessageAsync(
                    chatId: chatId,
                    text: aimlService.Talk(chatId, username, $"предсказываю {predicted}"),
                    cancellationToken: cancellationToken);
                await botClient.SendTextMessageAsync(
                    chatId: chatId,
                    text: aimlService.Talk(chatId, username, $"жду правду"),
                    cancellationToken: cancellationToken);
                lastRecognizedLetter = predicted;
                return;
            }

            if (message.Type == MessageType.Video)
            {
                await client.SendTextMessageAsync(message.Chat.Id, aimlService.Talk(chatId, username, "Видео"), cancellationToken: cancellationToken);
                return;
            }
            if (message.Type == MessageType.Audio)
            {
                await client.SendTextMessageAsync(message.Chat.Id, aimlService.Talk(chatId, username, "Аудио"), cancellationToken: cancellationToken);
                return;
            }
        }

        Task HandleErrorAsync(ITelegramBotClient botClient, Exception exception, CancellationToken cancellationToken)
        {
            var apiRequestException = exception as ApiRequestException;
            if (apiRequestException != null)
                Console.WriteLine($"Telegram API Error:\n[{apiRequestException.ErrorCode}]\n{apiRequestException.Message}");
            else
                Console.WriteLine(exception.ToString());
            return Task.CompletedTask;
        }

        public void Dispose()
        {
            // Заканчиваем работу и отменяем токен
            cts.Cancel();
        }
    }
}
