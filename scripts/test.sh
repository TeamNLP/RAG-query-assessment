# Pass the env-vars to MYCOMMAND
# eval $(egrep -v '^#' .env | xargs) MYCOMMAND
# … or ...
# Export the vars in .env into your shell:

#!/bin/sh




echo $LARGE_MODEL_NAME