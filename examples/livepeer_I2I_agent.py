"""This script demonstrates how to create a Image-to-Image (I2I) agent using
Livepeer.

The implementation relies on the experimental `livepeer_ai` client SDK, which
can be found at: https://github.com/rickstaa/livepeer-ai-sdks/tree/main/sdks/python

For detailed instructions on how to use this SDK, please refer to the README.md
file in the repository: https://github.com/rickstaa/livepeer-ai-sdks/blob/main/sdks/python/README.md
"""  # noqa: E501

import argparse
import requests
from PIL import Image
import io

import livepeer_ai
from livepeer_ai.rest import ApiException

from hive_agent import HiveAgent

configuration = livepeer_ai.Configuration(host="https://dream-gateway.livepeer.cloud")


class LivepeerI2IAgent:
    def __init__(self, configuration):
        self.configuration = configuration
        self.host = configuration.host

    def request_image_to_image(self, **kwargs):
        with livepeer_ai.ApiClient(self.configuration) as api_client:
            self.api_instance = livepeer_ai.DefaultApi(api_client)
            try:
                api_response = self.api_instance.image_to_image(**kwargs)
            except ApiException as e:
                raise Exception(e)
            return api_response

    def download_image(self, media):
        url = self.host + media.url
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open("I2I_output.png", "wb") as out_file:
                out_file.write(response.content)
            print("Image downloaded successfully")
        else:
            print("Unable to download the image")

    def show_image(self, image):
        image = Image.open(io.BytesIO(image))
        image.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Image-to-Image job using Livepeer AI subnet."
    )
    parser.add_argument(
        "--image", type=str, required=True, help="The input image file path"
    )
    args = parser.parse_args()
    image_path = args.image

    # Retrieve the image from the file path
    with open(image_path, "rb") as f:
        image = f.read()

    LivepeerI2IAgent = LivepeerI2IAgent(configuration)

    livepeer_T2I_agent = HiveAgent(
        name="LivepeerI2IAgent",
        functions=[LivepeerI2IAgent.request_image_to_image],
        instruction="Convert text to image using the Livepeer AI service",
        port=8002,
    )

    # Check if the agent is setup correctly
    test_response = LivepeerI2IAgent.request_image_to_image(
        model_id="timbrooks/instruct-pix2pix",
        prompt="A Cat on the Beach!",
        image=image,
    )

    # Download the first image from the response.
    image = LivepeerI2IAgent.download_image(test_response.images[0])

    # Show the image
    LivepeerI2IAgent.show_image(image)

    # TODO: Implement HiveAgent endpoints and logic.
    livepeer_T2I_agent.run()
