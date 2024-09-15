import os
import datetime
import numpy as np
from openai import OpenAI
from omegaconf import OmegaConf

if __name__ == "__main__":

    # Read config file
    config = OmegaConf.load("./configs/default.yaml")

    # Create log directory
    dt = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
    path_log = "./logs/" + config.name + "/" + dt + "/"
    os.makedirs(path_log, exist_ok=True)
    
    # Create OpenAI Client
    client = OpenAI(
        api_key=os.environ["MSC24_API_KEY"],
    )

    # Send prompt, retrieve response
    system_content = "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."
    user_content = "Compose a poem that explains the concept of recursion in programming."
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
    )

    # Print and save output
    output = completion.choices[0].message.content
    print(output)

    with open(path_log + "log.txt", "a") as fp:
        fp.write("### System ###\n" + system_content + "\n")
        fp.write("### User ###\n" + user_content + "\n")
        fp.write("### AI ###\n" + output + "\n")
