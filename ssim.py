from skimage.metrics import structural_similarity as ssim

def calculate_ssim(attn1, attn2):
    ssim_score = ssim(attn1, attn2)
    return ssim_score.item()
