"""
Download Wine Quality Data from Azure Blob Storage
"""

from azure.storage.blob import BlobServiceClient
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class AzureBlobDownloader:
    """
    Downloads data from Azure Blob Storage
    """
    
    def __init__(self, connection_string, container_name):
        """
        Initialize Azure Blob Storage client
        
        Args:
            connection_string: Azure Storage connection string
            container_name: Name of the blob container
        """
        self.connection_string = connection_string
        self.container_name = container_name
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)
    
    def download_file(self, blob_name, local_file_path):
        """
        Download a single blob to local file
        
        Args:
            blob_name: Name of the blob in Azure
            local_file_path: Where to save the file locally
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            print(f"Downloading {blob_name} -> {local_file_path}")
            
            # Create directory if it doesn't exist
            Path(local_file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(local_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            
            print(f"Downloaded successfully")
            return local_file_path
            
        except Exception as e:
            print(f"Error downloading {blob_name}: {e}")
            raise
    
    def download_all_csv_files(self, local_directory="data"):
        """
        Download all CSV files from the container
        
        Args:
            local_directory: Local directory to save files
        """
        os.makedirs(local_directory, exist_ok=True)
        
        print(f"Downloading CSV files to {local_directory}/")
        
        # List all blobs
        blobs = self.container_client.list_blobs()
        csv_blobs = [blob for blob in blobs if blob.name.endswith('.csv')]
        
        if not csv_blobs:
            print("No CSV files found in container")
            return []
        
        downloaded_files = []
        for blob in csv_blobs:
            local_path = os.path.join(local_directory, blob.name)
            self.download_file(blob.name, local_path)
            downloaded_files.append(local_path)
        
        return downloaded_files

def download_wine_data(connection_string=None, container_name="wine-quality-data", 
                       output_dir="data"):
    """
    Convenience function to download wine quality data
    
    Args:
        connection_string: Azure connection string (reads from env if None)
        container_name: Name of the blob container
        output_dir: Where to save the files locally
    
    Returns:
        List of downloaded file paths
    """
    
    # Try to get connection string from environment variable if not provided
    if connection_string is None:
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        
        if connection_string is None:
            raise ValueError(
                "No connection string provided. Either pass it as an argument "
                "or set the AZURE_STORAGE_CONNECTION_STRING environment variable"
            )
    
    downloader = AzureBlobDownloader(connection_string, container_name)
    downloaded_files = downloader.download_all_csv_files(output_dir)
    
    print(f"Downloaded {len(downloaded_files)} files")
    return downloaded_files

def main():
    """Main function for standalone use"""
    
    print("Azure Blob Storage Download Script")
    print("=" * 60)
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Configuration
    CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "wine-quality-data")
    OUTPUT_DIR = "data"
    
    if not CONNECTION_STRING:
        print("ERROR: AZURE_STORAGE_CONNECTION_STRING not found!")
        print("Setup instructions:")
        print("   1. Copy .env.example to .env:")
        print("      cp .env.example .env")
        print("   2. Edit .env and add your connection string")
        print("   3. Get your connection string from Azure Portal:")
        print("      - Go to your Storage Account")
        print("      - Click 'Access keys'")
        print("      - Copy the 'Connection string' value")
        return
    
    try:
        downloaded_files = download_wine_data(
            connection_string=CONNECTION_STRING,
            container_name=CONTAINER_NAME,
            output_dir=OUTPUT_DIR
        )
        
        print("\n" + "=" * 60)
        print("Download complete!")
        print(f"Files saved in '{OUTPUT_DIR}/' directory:")
        for file_path in downloaded_files:
            print(f"   - {file_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()