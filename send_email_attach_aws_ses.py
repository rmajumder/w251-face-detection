import os
import boto3
from botocore.exceptions import ClientError
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart

SENDER = "Rishi <xxx@berkeley.edu>"
RECIPIENT = "xxxxx"
AWS_REGION = "xxxxx"
ACCESS_KEY = "xxxxx"
SECRET_KEY = "xxxxx"

CHARSET = "UTF-8"
client = boto3.client('ses',region_name=AWS_REGION, aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

def send_email(img_path, subject, body):

    SUBJECT = subject
    # The full path to the file that will be attached to the email.
    ATTACHMENT = img_path

    # The email body for recipients with non-HTML email clients.
    BODY_TEXT = body

    # The HTML body of the email.
    BODY_HTML = """\
    <html>
    <head></head>
    <body>
    <h1>Unidentified Person</h1>
    </body>
    </html>
    """

    # Create a multipart/mixed parent container.
    msg = MIMEMultipart('mixed')
    # Add subject, from and to lines.
    msg['Subject'] = SUBJECT 
    msg['From'] = SENDER 
    msg['To'] = RECIPIENT

    # Create a multipart/alternative child container.
    msg_body = MIMEMultipart('alternative')

    # Encode the text and HTML content and set the character encoding. This step is
    # necessary if you're sending a message with characters outside the ASCII range.
    textpart = MIMEText(BODY_TEXT.encode(CHARSET), 'plain', CHARSET)
    htmlpart = MIMEText(BODY_HTML.encode(CHARSET), 'html', CHARSET)

    # Add the text and HTML parts to the child container.
    msg_body.attach(textpart)
    msg_body.attach(htmlpart)

    # Define the attachment part and encode it using MIMEApplication.
    att = MIMEApplication(open(ATTACHMENT, 'rb').read())

    # Add a header to tell the email client to treat this part as an attachment,
    # and to give the attachment a name.
    att.add_header('Content-Disposition','attachment',filename=os.path.basename(ATTACHMENT))

    # Attach the multipart/alternative child container to the multipart/mixed
    # parent container.
    msg.attach(msg_body)

    # Add the attachment to the parent container.
    msg.attach(att)

    # Try to send the email.
    try:
         #Provide the contents of the email.
        response = client.send_raw_email(
            Source=SENDER,
            Destinations=[
                RECIPIENT
            ],
            RawMessage={
                'Data':msg.as_string(),
            }
        )
    # Display an error if something goes wrong.	
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Email sent! Message ID:"),
        print(response['MessageId'])
