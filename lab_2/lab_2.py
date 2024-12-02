######################################################################################################################
#  Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.                                           #
#                                                                                                                    #
#  Licensed under the Apache License, Version 2.0 (the License). You may not use this file except in compliance    #
#  with the License. A copy of the License is located at                                                             #
#                                                                                                                    #
#      http://www.apache.org/licenses/LICENSE-2.0                                                                    #
#                                                                                                                    #
#  or in the 'license' file accompanying this file. This file is distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES #
#  OR CONDITIONS OF ANY KIND, express or implied. See the License for the specific language governing permissions    #
#  and limitations under the License.                                                                                #
#####################################################################################################################

import boto3
import json
import os
import copy
import pandas as pd
import numpy as np
import urllib.parse
from urllib.parse import unquote
from botocore.client import Config
import time
import logging
import datetime
import traceback
from botocore.exceptions import ClientError
from botocore.client import Config

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    print("Received event: " + json.dumps(event))

    for record in event['Records']:
        message = json.loads(record['Sns']['Message'])
        print("Message: {}".format(message))

        request = {}

        request["jobId"] = message['JobId']
        request["jobTag"] = message['JobTag']
        request["jobStatus"] = message['Status']
        request["jobAPI"] = message['API']
        request["bucketName"] = message['DocumentLocation']['S3Bucket']
        request["objectName"] = message['DocumentLocation']['S3ObjectName']
        request["outputBucketName"] = os.environ['OUTPUT_BUCKET']

        print("Full path of input file is {}/{}".format(request["bucketName"], request["objectName"]))

        processRequest(request)


def getJobResults(api, jobId):
    texttract_client = getClient('textract', 'us-east-1')
    blocks = []
    analysis = {}
    response = texttract_client.get_document_analysis(
        JobId=jobId
    )
    analysis = copy.deepcopy(response)
    while True:
        for block in response["Blocks"]:
            blocks.append(block)
        if ("NextToken" not in response.keys()):
            break
        next_token = response["NextToken"]
        response = texttract_client.get_document_analysis(
            JobId=jobId,
            NextToken=next_token
        )
    analysis.pop("NextToken", None)
    analysis["Blocks"] = blocks

    Total_Pages = response['DocumentMetadata']['Pages']
    finalJSON_allpage = []
    print(Total_Pages)
    for i in range(Total_Pages):
        thisPage = i + 1
        thisPage_json = parsejson_inorder_perpage(analysis, thisPage)
        finalJSON_allpage.append({'Page': thisPage, 'Content': thisPage_json})
        print(f"Page {thisPage} parsed")

    # blocks = []
    # analysis = {}
    # pages = []
    # thisPage = 1
    # finalJSON_allpage=[]
    # time.sleep(5)

    # client = getClient('textract', 'us-east-1')
    # response = client.get_document_analysis(JobId=jobId)
    # analysis = copy.deepcopy(response)

    # pages.append(response)
    # print("Resultset page recieved: {}".format(len(pages)))
    # nextToken = None
    # if('NextToken' in response):
    #     nextToken = response['NextToken']
    #     print("Next token: {}".format(nextToken))

    # thisPage_json=parsejson_inorder_perpage(response,thisPage)

    # finalJSON_allpage.append({'Page':thisPage,'Content':thisPage_json})
    # print("Page {} json is {}".format(thisPage, response))

    # while(nextToken):
    #     try:
    #         time.sleep(5)
    #         response = client.get_document_analysis(JobId=jobId, NextToken=nextToken)

    #         pages.append(response)
    #         print("Resultset page recieved: {}".format(response))
    #         nextToken = None
    #         if('NextToken' in response):
    #             nextToken = response['NextToken']
    #             print("Next token: {}".format(nextToken))

    #         thisPage=thisPage+1

    #         thisPage_json=parsejson_inorder_perpage(response,thisPage)

    #         finalJSON_allpage.append({'Page':thisPage,'Content':thisPage_json})
    #         print("Page {} json is {}".format(thisPage, response))

    #     except Exception as e:
    #         if(e.__class__.__name__ == 'ProvisionedThroughputExceededException'):
    #             print("ProvisionedThroughputExceededException.")
    #             print("Waiting for few seconds...")
    #             time.sleep(5)
    #             print("Waking up...")

    # print('The resulting json is {}'.format(finalJSON_allpage))

    return finalJSON_allpage


def processRequest(request):
    s3_client = getClient('s3', 'us-east-1')

    output = ""

    print("Request : {}".format(request))

    jobId = request['jobId']
    documentId = request['jobTag']
    jobStatus = request['jobStatus']
    jobAPI = request['jobAPI']
    bucketName = request['bucketName']
    outputBucketName = request['outputBucketName']
    objectName = request['objectName']

    directory = objectName.split('/')

    upload_path = ''
    for subdirectory in directory:
        if subdirectory != directory[-1]:
            upload_path += (subdirectory + '/')

    file_name = directory[-1]

    file_name_no_ext = file_name.rsplit(".", 1)[0]

    upload_path = upload_path + file_name_no_ext + '/textract/'

    finalJSON_allpage = getJobResults(jobAPI, jobId)

    analyses_bucket_name = outputBucketName
    analyses_bucket_key = "{}".format(objectName.replace('.PDF', '.json'))
    s3_client.put_object(
        Bucket=analyses_bucket_name,
        Key=upload_path + analyses_bucket_key,
        Body=json.dumps(finalJSON_allpage).encode('utf-8')
    )

    _writeToDynamoDB("pdf-to-json", objectName, bucketName + '/' + objectName, finalJSON_allpage)

    return {
        'statusCode': 200,
        'body': json.dumps(finalJSON_allpage)
    }


def find_value_block(key_block, value_map):
    for relationship in key_block['Relationships']:
        if relationship['Type'] == 'VALUE':
            for value_id in relationship['Ids']:
                value_block = value_map[value_id]
    return value_block


def get_text(result, blocks_map):
    text = ''
    if 'Relationships' in result:
        for relationship in result['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    word = blocks_map[child_id]
                    if word['BlockType'] == 'WORD':
                        text += word['Text'] + ' '
                    if word['BlockType'] == 'SELECTION_ELEMENT':
                        if word['SelectionStatus'] == 'SELECTED':
                            text += 'X '

    return text


def find_Key_value_inrange(response, top, bottom, thisPage):
    # given Textract Response, and [top,bottom] - bounding box need to search for
    # find Key:value pairs within the bounding box

    # get key_map,value_map,block_map from response (textract JSON)

    blocks = response['Blocks']
    key_map = {}
    value_map = {}
    block_map = {}
    for block in blocks:
        if block['Page'] == thisPage:
            block_id = block['Id']
            block_map[block_id] = block
            if block['BlockType'] == "KEY_VALUE_SET" or block['BlockType'] == 'KEY' or block['BlockType'] == 'VALUE':
                if 'KEY' in block['EntityTypes']:
                    key_map[block_id] = block
                else:
                    value_map[block_id] = block

    ## find key-value pair within given bounding box:
    kv_pair = {}
    for block_id, key_block in key_map.items():
        value_block = find_value_block(key_block, value_map)
        key = get_text(key_block, block_map)
        val = get_text(value_block, block_map)
        if (value_block['Geometry']['BoundingBox']['Top'] >= top and
                value_block['Geometry']['BoundingBox']['Top'] + value_block['Geometry']['BoundingBox'][
                    'Height'] <= bottom):
            kv_pair[key] = val
    return kv_pair


def get_rows_columns_map(table_result, blocks_map):
    rows = {}
    for relationship in table_result['Relationships']:
        if relationship['Type'] == 'CHILD':
            for child_id in relationship['Ids']:
                cell = blocks_map[child_id]
                if cell['BlockType'] == 'CELL':
                    row_index = cell['RowIndex']
                    col_index = cell['ColumnIndex']
                    if row_index not in rows:
                        rows[row_index] = {}
                    rows[row_index][col_index] = get_text(cell, blocks_map)
    return rows


def get_tables_fromJSON_inrange(response, top, bottom, thisPage):
    # given respones and top/bottom corrdinate, return tables in the range
    blocks = response['Blocks']
    blocks_map = {}
    table_blocks = []
    for block in blocks:
        if block['Page'] == thisPage:
            blocks_map[block['Id']] = block
            if block['BlockType'] == "TABLE":
                if (block['Geometry']['BoundingBox']['Top'] >= top and
                        block['Geometry']['BoundingBox']['Top'] +
                        block['Geometry']['BoundingBox']['Height'] <= bottom):
                    table_blocks.append(block)

    if len(table_blocks) <= 0:
        return

    AllTables = []
    for table_result in table_blocks:
        tableMatrix = []
        rows = get_rows_columns_map(table_result, blocks_map)
        for row_index, cols in rows.items():
            thisRow = []
            for col_index, text in cols.items():
                thisRow.append(text)
            tableMatrix.append(thisRow)
        AllTables.append(tableMatrix)
    return AllTables


### get tables coordinate in range:
def get_tables_coord_inrange(response, top, bottom, thisPage):
    # given respones and top/bottom corrdinate, return tables in the range
    blocks = response['Blocks']
    blocks_map = {}
    table_blocks = []
    for block in blocks:
        if block['Page'] == thisPage:
            blocks_map[block['Id']] = block
            if block['BlockType'] == "TABLE":
                if (block['Geometry']['BoundingBox']['Top'] >= top and
                        block['Geometry']['BoundingBox']['Top'] +
                        block['Geometry']['BoundingBox']['Height'] <= bottom):
                    table_blocks.append(block)

    if len(table_blocks) <= 0:
        return

    AllTables_coord = []
    for table_result in table_blocks:
        AllTables_coord.append(table_result['Geometry']['BoundingBox'])
    return AllTables_coord


def box_within_box(box1, box2):
    # check if bounding box1 is completely within bounding box2
    # box1:{Width,Height,Left,Top}
    # box2:{Width,Height,Left,Top}
    if box1['Top'] >= box2['Top'] and box1['Left'] >= box2['Left'] and box1['Top'] + box1['Height'] <= box2['Top'] + \
            box2['Height'] and box1['Left'] + box1['Width'] <= box2['Left'] + box2['Width']:
        return True
    else:
        return False


def find_Key_value_inrange_notInTable(response, top, bottom, thisPage):
    # given Textract Response, and [top,bottom] - bounding box need to search for
    # find Key:value pairs within the bounding box

    # get key_map,value_map,block_map from response (textract JSON)
    blocks = response['Blocks']
    key_map = {}
    value_map = {}
    block_map = {}
    for block in blocks:
        if block['Page'] == thisPage:
            block_id = block['Id']
            block_map[block_id] = block
            if block['BlockType'] == "KEY_VALUE_SET" or block['BlockType'] == 'KEY' or block['BlockType'] == 'VALUE':
                if 'KEY' in block['EntityTypes']:
                    key_map[block_id] = block
                else:
                    value_map[block_id] = block

    # get all table coordicates in range:
    AllTables_coord = get_tables_coord_inrange(response, top, bottom, thisPage)

    ## find key-value pair within given bounding box:
    kv_pair = {}
    for block_id, key_block in key_map.items():
        value_block = find_value_block(key_block, value_map)
        key = get_text(key_block, block_map)
        val = get_text(value_block, block_map)
        if (value_block['Geometry']['BoundingBox']['Top'] >= top and
                value_block['Geometry']['BoundingBox']['Top'] + value_block['Geometry']['BoundingBox'][
                    'Height'] <= bottom):

            kv_overlap_table_list = []
            if AllTables_coord is not None:
                for table_coord in AllTables_coord:
                    kv_overlap_table_list.append(box_within_box(value_block['Geometry']['BoundingBox'], table_coord))
                if sum(kv_overlap_table_list) == 0:
                    kv_pair[key] = val
            else:
                kv_pair[key] = val
    return kv_pair


### function: take response of multi-page Textract, and page_number
### return order sequence JSON for that page Text1->KV/Table->Text2->KV/Table..
def parsejson_inorder_perpage(response, thisPage):
    # input: response - multipage Textract response JSON
    #        thisPage - page number : 1,2,3..
    # output: clean parsed JSON for this Page in correct order
    TextList = []
    ID_list_KV_Table = []
    for block in response['Blocks']:
        if block['Page'] == thisPage:
            if block['BlockType'] == 'TABLE' or block['BlockType'] == 'CELL' or \
                    block['BlockType'] == 'KEY_VALUE_SET' or block['BlockType'] == 'KEY' or block[
                'BlockType'] == 'VALUE' or \
                    block['BlockType'] == 'SELECTION_ELEMENT':

                kv_id = block['Id']
                if kv_id not in ID_list_KV_Table:
                    ID_list_KV_Table.append(kv_id)

                child_idlist = []
                if 'Relationships' in block.keys():
                    for child in block['Relationships']:
                        child_idlist.append(child['Ids'])
                    flat_child_idlist = [item for sublist in child_idlist for item in sublist]
                    for childid in flat_child_idlist:
                        if childid not in ID_list_KV_Table:
                            ID_list_KV_Table.append(childid)
    TextList = []
    for block in response['Blocks']:
        if block['Page'] == thisPage:
            if block['BlockType'] == 'LINE':

                thisline_idlist = []
                thisline_idlist.append(block['Id'])
                child_idlist = []
                if 'Relationships' in block.keys():
                    for child in block['Relationships']:
                        child_idlist.append(child['Ids'])
                    flat_child_idlist = [item for sublist in child_idlist for item in sublist]
                    for childid in flat_child_idlist:
                        thisline_idlist.append(childid)

                setLineID = set(thisline_idlist)
                setAllKVTableID = set(ID_list_KV_Table)
                if len(setLineID.intersection(setAllKVTableID)) == 0:
                    #           print(block['Text'])
                    thisDict = {'Line': block['Text'],
                                'Left': block['Geometry']['BoundingBox']['Left'],
                                'Top': block['Geometry']['BoundingBox']['Top'],
                                'Width': block['Geometry']['BoundingBox']['Width'],
                                'Height': block['Geometry']['BoundingBox']['Height']}
                    #           print(thisDict)
                    TextList.append(thisDict)

    finalJSON = []
    for i in range(len(TextList) - 1):
        thisText = TextList[i]['Line']
        thisTop = TextList[i]['Top']
        thisBottom = TextList[i + 1]['Top'] + TextList[i + 1]['Height']
        #           thisText_KV=find_Key_value_inrange_notInTable(response,thisTop,thisBottom,thisPage)
        thisText_KV = find_Key_value_inrange(response, thisTop, thisBottom, thisPage)
        thisText_Table = get_tables_fromJSON_inrange(response, thisTop, thisBottom, thisPage)
        finalJSON.append({thisText: {'KeyValue': thisText_KV, 'Tables': thisText_Table}})

    if (len(TextList) > 0):
        ## last line Text to bottom of page:
        lastText = TextList[len(TextList) - 1]['Line']
        lastTop = TextList[len(TextList) - 1]['Top']
        lastBottom = 1
        #       thisText_KV=find_Key_value_inrange_notInTable(response,lastTop,lastBottom,thisPage)
        thisText_KV = find_Key_value_inrange(response, lastTop, lastBottom, thisPage)
        thisText_Table = get_tables_fromJSON_inrange(response, lastTop, lastBottom, thisPage)
        finalJSON.append({lastText: {'KeyValue': thisText_KV, 'Tables': thisText_Table}})

    return finalJSON


def _writeToDynamoDB(dd_table_name, Id, fullFilePath, fullPdfJson):
    # Get the service resource.
    dynamodb = getResource('dynamodb')
    dynamodb_client = getClient('dynamodb')

    dd_table_name = dd_table_name \
        .replace(" ", "-") \
        .replace("(", "-") \
        .replace(")", "-") \
        .replace("&", "-") \
        .replace(",", " ") \
        .replace(":", "-") \
        .replace('/', '--') \
        .replace("#", 'No') \
        .replace('"', 'Inch')

    if len(dd_table_name) <= 3:
        dd_table_name = dd_table_name + '-xxxx'

    print("DynamoDB table name is {}".format(dd_table_name))

    # Create the DynamoDB table.
    try:

        existing_tables = list([x.name for x in dynamodb.tables.all()])

        # existing_tables = dynamodb_client.list_tables()['TableNames']

        if dd_table_name not in existing_tables:
            table = dynamodb.create_table(
                TableName=dd_table_name,
                KeySchema=[
                    {
                        'AttributeName': 'Id',
                        'KeyType': 'HASH'
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'Id',
                        'AttributeType': 'S'
                    },
                ],
                BillingMode='PAY_PER_REQUEST',
            )
            # Wait until the table exists, this will take a minute or so
            table.meta.client.get_waiter('table_exists').wait(TableName=dd_table_name)
            # Print out some data about the table.
            print("Table successfully created. Item count is: " + str(table.item_count))
    # except dynamodb_client.exceptions.ResourceInUseException:
    except ClientError as e:
        if e.response['Error']['Code'] in ["ThrottlingException", "ProvisionedThroughputExceededException"]:
            msg = f"DynamoDB ] Write Failed from DynamoDB, Throttling Exception [{e}] [{traceback.format_exc()}]"
            logging.warning(msg)
            raise e
        else:
            msg = f"DynamoDB Write Failed from DynamoDB Exception [{e}] [{traceback.format_exc()}]"
            logging.error(msg)
            raise e

    except Exception as e:
        msg = f"DynamoDB Write Failed from DynamoDB Exception [{e}] [{traceback.format_exc()}]"
        logging.error(msg)
        raise Exception(e)

    table = dynamodb.Table(dd_table_name)

    try:
        table.put_item(Item=
        {
            'Id': Id,
            'FilePath': fullFilePath,
            'PdfJsonRegularFormat': str(fullPdfJson),
            'PdfJsonDynamoFormat': fullPdfJson,
            'DateTime': datetime.datetime.utcnow().isoformat(),
        }
        )
    except ClientError as e:
        if e.response['Error']['Code'] in ["ThrottlingException", "ProvisionedThroughputExceededException"]:
            msg = f"DynamoDB ] Write Failed from DynamoDB, Throttling Exception [{e}] [{traceback.format_exc()}]"
            logging.warning(msg)
            raise e

        else:
            msg = f"DynamoDB Write Failed from DynamoDB Exception [{e}] [{traceback.format_exc()}]"
            logging.error(msg)
            raise e

    except Exception as e:
        msg = f"DynamoDB Write Failed from DynamoDB Exception [{e}] [{traceback.format_exc()}]"
        logging.error(msg)
        raise Exception(e)


def dict_to_item(raw):
    if type(raw) is dict:
        resp = {}
        for k, v in raw.items():
            if type(v) is str:
                resp[k] = {
                    'S': v
                }
            elif type(v) is int:
                resp[k] = {
                    'I': str(v)
                }
            elif type(v) is dict:
                resp[k] = {
                    'M': dict_to_item(v)
                }
            elif type(v) is list:
                resp[k] = []
                for i in v:
                    resp[k].append(dict_to_item(i))

        return resp
    elif type(raw) is str:
        return {
            'S': raw
        }
    elif type(raw) is int:
        return {
            'I': str(raw)
        }


def getClient(name, awsRegion=None):
    config = Config(
        retries=dict(
            max_attempts=30
        )
    )
    if (awsRegion):
        return boto3.client(name, region_name=awsRegion, config=config)
    else:
        return boto3.client(name, config=config)


def getResource(name, awsRegion=None):
    config = Config(
        retries=dict(
            max_attempts=30
        )
    )

    if (awsRegion):
        return boto3.resource(name, region_name=awsRegion, config=config)
    else:
        return boto3.resource(name, config=config)