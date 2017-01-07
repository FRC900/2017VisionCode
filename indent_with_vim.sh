#!/bin/bash
vim -c "normal ggVG=" -e $1 <<'EOF'
:wq
EOF
