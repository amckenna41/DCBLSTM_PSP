FILE="psp_gcp/config/dcblstm.json"
# basename "$FILE"
f="$(basename -- $FILE)"
echo "$f"

filename="${f%.*}"
echo $filename

# basename psp_gcp/config/dcblstm.json
