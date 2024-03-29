import model

# Create an instance of your model
my_model = model.CSDN_Tem(in_ch=3, out_ch=64)
DCE_net = model.enhance_net_nopool(1).cuda()
# Now you can print the instance of the model
total_params = sum(p.numel() for p in DCE_net.parameters())

print(f"Total Trainable Parameters: {total_params}")
