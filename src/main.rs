extern crate tensorflow;

use std::error::Error;

use nokhwa::{
    pixel_format::RgbFormat,
    utils::{
        CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution,
    },
    Camera,
};
use tensorflow::{
    eager::{raw_ops, Context, ContextOptions},
    DataType, Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor,
};

const IMG_W: u32 = 640;
const IMG_H: u32 = 480;
const FPS: u32 = 30;

fn main() -> Result<(), Box<dyn Error>> {
    let mut camera = Camera::new(
        CameraIndex::Index(0),
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(CameraFormat::new(
            Resolution::new(IMG_W, IMG_H),
            FrameFormat::MJPEG,
            FPS,
        ))),
    )
    .unwrap();

    let model_dir = "movenet";
    let mut graph = Graph::new();
    let bundle = SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, model_dir)?;
    let session = &bundle.session;
    let op_in = graph.operation_by_name_required("serving_default_input")?;
    let op_out = graph.operation_by_name_required("StatefulPartitionedCall")?;
    let context = Context::new(ContextOptions::new())?;

    camera.open_stream().unwrap();
    loop {
        let frame = camera.frame();
        match frame {
            Ok(f) => {
                // let buf = f.decode_image::<RgbFormat>().unwrap();
                let img;
                unsafe {
                    let buf = String::from_utf8_unchecked(f.buffer_bytes().into()); // TODO does this have to explicitly be a 0D tensor or does it take care of that for us?
                    img =
                        raw_ops::expand_dims(&context, &raw_ops::decode_jpeg(&context, &buf)?, &0)?;
                }
                // let mut mn_in = rgb_buf_to_tensor(buf.into_raw(), IMG_W as u64, IMG_H as u64);
                let boxes = Tensor::new(&[1, 4])
                    .with_values(&[
                        0_f32,
                        0.5 - IMG_H as f32 / (2.0 * IMG_W as f32),
                        1.0,
                        0.5 + IMG_H as f32 / (2.0 * IMG_W as f32),
                    ])?
                    .into_handle(&context)?;
                let box_ind = Tensor::new(&[1])
                    .with_values(&[0_i32])?
                    .into_handle(&context)?;
                let crop_size = Tensor::new(&[2])
                    .with_values(&[256_i32, 256])?
                    .into_handle(&context)?;
                let mn_out: Tensor<f32> = unsafe {
                    let mn_in: Tensor<i32> = raw_ops::Cast::new()
                        .DstT(DataType::Int32)
                        .call(
                            &context,
                            &raw_ops::crop_and_resize(
                                &context, &img, &boxes, &box_ind, &crop_size,
                            )?,
                        )?
                        .resolve()?
                        .into_tensor();
                    let mut args = SessionRunArgs::new();
                    args.add_feed(&op_in, 0, &mn_in);
                    let f_tok = args.request_fetch(&op_out, 0);
                    session.run(&mut args)?;
                    args.fetch(f_tok)?
                };
                println!(
                    "y: {}\nx: {}",
                    mn_out.get(&[0, 0, 0, 0]),
                    mn_out.get(&[0, 0, 0, 1])
                );
            }
            Err(e) => {
                println!("{:?}", e);
            }
        }
    }
    Ok(())
}

fn rgb_buf_to_tensor(img: Vec<u8>, w: u64, h: u64) -> Tensor<i32> {
    let v: Vec<i32> = img.into_iter().map(|x| x as i32).collect();
    Tensor::new(&[w, h, 3]).with_values(&v).unwrap()
}
