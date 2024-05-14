"""This script demonstrates how to create a Text-to-Image (T2I) agent using
Livepeer.

The implementation relies on the experimental `livepeer_ai` client SDK, which
can be found at: https://github.com/rickstaa/livepeer-ai-sdks/tree/main/sdks/python

For detailed instructions on how to use this SDK, please refer to the README.md
file in the repository: https://github.com/rickstaa/livepeer-ai-sdks/blob/main/sdks/python/README.md
"""  # noqa: E501

import requests
from PIL import Image
import io

import livepeer_ai
from livepeer_ai.rest import ApiException

from hive_agent import HiveAgent

configuration = livepeer_ai.Configuration(host="https://dream-gateway.livepeer.cloud")


class LivepeerT2IAgent:
    def __init__(self, configuration):
        self.configuration = configuration
        self.host = configuration.host

    def request_text_to_image(self, **kwargs):
        with livepeer_ai.ApiClient(self.configuration) as api_client:
            self.api_instance = livepeer_ai.DefaultApi(api_client)
            try:
                api_response = self.api_instance.text_to_image(**kwargs)
            except ApiException as e:
                raise Exception(e)
            return api_response

    def download_image(self, media):
        url = self.host + media.url
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open("image.png", "wb") as out_file:
                out_file.write(response.content)
            print("Image downloaded successfully")

            # Display the image
            image = Image.open(io.BytesIO(response.content))
            image.show()
        else:
            print("Unable to download the image")


if __name__ == "__main__":
    LivepeerT2IAgent = LivepeerT2IAgent(configuration)

    livepeer_T2I_agent = HiveAgent(
        name="LivepeerT2IAgent",
        functions=[LivepeerT2IAgent.request_text_to_image],
        instruction="Convert text to image using the Livepeer AI service",
        port=8001,
    )

    # Check if the agent is setup correctly
    test_response = LivepeerT2IAgent.request_text_to_image(
        text_to_image_params=livepeer_ai.TextToImageParams(
            model_id="ByteDance/SDXL-Lightning", prompt="A Cat on the Beach!"
        )
    )

    # Download the first image from the response.
    image = LivepeerT2IAgent.download_image(test_response.images[0])

    # TODO: Implement HiveAgent endpoints and logic.
    livepeer_T2I_agent.run()
