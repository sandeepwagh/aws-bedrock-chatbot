import json
import boto3

bedrock = boto3.client("bedrock-runtime")

def lambda_handler(event, context):
    try:
        # Parse user input
        body = json.loads(event["body"])
        user_input = body.get("message", "")

        # Prepare request to Titan Text Lite
        titan_input = {
            "inputText": f"Respond like a chatbot to: {user_input}"
        }

        response = bedrock.invoke_model(
            modelId="amazon.titan-text-lite-v1",
            body=json.dumps(titan_input),
            contentType="application/json",
            accept="application/json"
        )

        model_output = json.loads(response["body"].read())

        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",  # Needed for S3 site
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "reply": model_output["results"][0]["outputText"]
            })
        }

    except Exception as e:
        print(f"Error: {e}")
        return {
            "statusCode": 500,
            "headers": {
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": "Something went wrong"})
        }
