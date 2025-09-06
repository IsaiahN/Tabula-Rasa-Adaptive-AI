# Ensure ARC_API_KEY is set in your environment or .env file before running this script.
if (-not $env:ARC_API_KEY) {
	Write-Host "ERROR: ARC_API_KEY environment variable is not set. Please set it in your environment or .env file."
	exit 1
}
Start-Process -NoNewWindow -FilePath 'C:/Users/Admin/AppData/Local/Microsoft/WindowsApps/python3.13.exe' -ArgumentList @('master_arc_trainer.py','--dashboard','gui','--mode','continuous-training','--max-cycles','100','--session-duration','240') -RedirectStandardOutput 'master_arc_trainer_output.log' -RedirectStandardError 'master_arc_trainer_error.log' -PassThru
