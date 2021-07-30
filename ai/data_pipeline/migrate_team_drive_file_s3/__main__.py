from ai.data_pipeline.migrate_team_drive_file_s3 import app

import argparse

if __name__ == '__main__':
    # Initiate the parser
    parser = argparse.ArgumentParser()

    # Add long and short argument
    parser.add_argument("--s3_bucket", "-b", help="Bucket Name", required=True)
    parser.add_argument("--s3_bucket_path", "-bp", help="Bucket Path", required=True)
    parser.add_argument("--file_name", "-f", help="Name of file on team drive.", required=True)
    parser.add_argument("--gdrive_id", "-g", help="Team Drive ID.", required=True)
    parser.add_argument("--log", "-l", help="Level of logs", default='INFO')

    app(parser)
