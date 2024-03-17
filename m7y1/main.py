import discord
from discord.ext import commands

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='$', intents=intents)

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command()
async def hello(ctx):
    await ctx.send(f'Hi! I am a bot {bot.user}!')

@bot.command()
async def heh(ctx, count_heh = 5):
    await ctx.send("he" * count_heh)

@bot.command()
async def test(ctx):
    files = ctx.message.attachments
    await ctx.send(files)
    await ctx.send(files[0].filename)
    await ctx.send(files[0].url)
    syf = files[0].filename.split(".")[-1]
    await files[0].save(f'file.{syf}')

def get_class(image, model, classes):
  from keras.models import load_model  # TensorFlow is required for Keras to work
  from PIL import Image, ImageOps  # Install pillow instead of PIL
  import numpy as np

  # Disable scientific notation for clarity
  np.set_printoptions(suppress=True)

  # Load the model
  model = load_model("keras_model.h5", compile=False)

  # Load the labels
  class_names = open("labels.txt", "r").readlines()

  # Create the array of the right shape to feed into the keras model
  # The 'length' or number of images you can put into the array is
  # determined by the first position in the shape tuple, in this case 1
  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

  # Replace this with the path to your image
  image = Image.open(image).convert("RGB")

  # resizing the image to be at least 224x224 and then cropping from the center
  size = (224, 224)
  image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

  # turn the image into a numpy array
  image_array = np.asarray(image)

  # Normalize the image
  normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

  # Load the image into the array
  data[0] = normalized_image_array

  # Predicts the model
  prediction = model.predict(data)
  index = np.argmax(prediction)
  class_name = class_names[index]
  confidence_score = prediction[0][index]

  # Print prediction and confidence score
  print("Class:", class_name[2:], end="")
  print("Confidence Score:", confidence_score)
  return index

@bot.command()
async def predict(ctx):
    files = ctx.message.attachments
    syf = files[0].filename.split(".")[-1]
    if syf != 'jpg' and syf != 'jpeg' and syf != 'png':
        await ctx.send("Неверный формат изображения!")
        return
    
    await files[0].save(f'file.{syf}')
    indx = get_class(f'file.{syf}','keras_model.h5', 'labels.txt')
    
    if indx == 0:
        await ctx.send("Это Сфинкс!")
    elif indx == 1:
        await ctx.send("Это Британская кошка!")
    elif indx == 2:
        await ctx.send("Это Сибирская кошка!")
    elif indx == 3:
        await ctx.send("Это Бурма!")

bot.run("MTE1NTQ0MzI4ODA5NTA4MDQ3OA.G7ZmZT.qPmIUxz6EABFGJf2GmfowRiZxtSSh3goMHtl14")