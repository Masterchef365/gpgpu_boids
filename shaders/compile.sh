for x in *.comp *.frag *.vert; do
    glslc -O $x -o $x.spv &
done
wait
